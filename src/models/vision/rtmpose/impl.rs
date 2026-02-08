use aksr::Builder;
use anyhow::Result;
use ndarray::s;
use rayon::prelude::*;
use std::f32::consts::PI;

use crate::{
    Config, DynConf, Engine, Engines, FromConfig, Hbb, Image, ImageProcessor, Keypoint, Model,
    Module, XAny, Xs, X, Y,
};

struct CentersAndScales {
    pub centers: Vec<Keypoint>,
    pub scales: Vec<Keypoint>,
}

/// RTMPose: Real-Time Multi-Person Pose Estimation
#[derive(Builder, Debug)]
pub struct RTMPose {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub spec: String,
    pub processor: ImageProcessor,
    pub nk: usize,
    pub kconfs: DynConf,
    pub names: Vec<String>,
    pub simcc_split_ratio: f32,
}

impl Model for RTMPose {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let spec = engine.spec().to_string();
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&256.into()).opt(),
            engine.try_width().unwrap_or(&192.into()).opt(),
        );
        let nk = config.inference.num_keypoints.unwrap_or(17);
        let kconfs = DynConf::new_or_default(&config.inference.keypoint_confs, nk);
        let names = config.inference.keypoint_names;
        let simcc_split_ratio = 2.0;
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            height,
            width,
            batch,
            spec,
            processor,
            nk,
            kconfs,
            names,
            simcc_split_ratio,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, input: Self::Input<'_>) -> Result<Vec<Y>> {
        let images = input;
        let (xs, centers_and_scales) =
            crate::perf!("RTMPose::preprocess", self.preprocess(images)?);
        let ys = crate::perf!("RTMPose::inference", engines.run(&Module::Model, &xs)?);
        let y = crate::perf!("RTMPose::postprocess", {
            self.postprocess(&ys, centers_and_scales)?
        });

        Ok(y)
    }
}

impl RTMPose {
    fn preprocess(&mut self, images: &[Image]) -> Result<(XAny, CentersAndScales)> {
        let model_input_size = (self.width as i32, self.height as i32);
        let results: Vec<(Image, Keypoint, Keypoint)> = images
            .par_iter()
            .map(|img| {
                let hbb = Hbb::from_xyxy(0.0, 0.0, img.width() as f32, img.height() as f32);
                let (center, scale) = Self::hbb2cs(&hbb, 1.25);
                let (resized_img, scale) =
                    Self::top_down_affine(model_input_size, &scale, &center, img)?;
                Ok((resized_img, center, scale))
            })
            .collect::<Result<Vec<_>>>()?;
        let (processed_images, centers, scales): (Vec<_>, Vec<_>, Vec<_>) =
            results.into_iter().fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut imgs, mut ctrs, mut scls), (img, ctr, scl)| {
                    imgs.push(img);
                    ctrs.push(ctr);
                    scls.push(scl);
                    (imgs, ctrs, scls)
                },
            );
        let x = self.processor.process(&processed_images)?;
        self.batch = processed_images.len();

        Ok((x, CentersAndScales { centers, scales }))
    }

    fn postprocess(
        &mut self,
        outputs: &Xs,
        centers_and_scales: CentersAndScales,
    ) -> Result<Vec<Y>> {
        let x0 = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output 0"))?;
        let x1 = outputs
            .get::<f32>(1)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output 1"))?;
        let simcc_x_array = X::from(x0);
        let simcc_y_array = X::from(x1);

        self.nk = simcc_x_array.shape()[1];
        let total_crops = simcc_x_array.shape()[0];
        let has_names = !self.names.is_empty();

        let results: Vec<Y> = (0..total_crops)
            .into_par_iter()
            .map(|batch_idx| {
                let center = &centers_and_scales.centers[batch_idx];
                let scale = &centers_and_scales.scales[batch_idx];

                let x_factor = scale.x() / (self.simcc_split_ratio * self.width as f32);
                let y_factor = scale.y() / (self.simcc_split_ratio * self.height as f32);
                let x_offset = center.x() - scale.x() * 0.5;
                let y_offset = center.y() - scale.y() * 0.5;

                let keypoints: Vec<Keypoint> = (0..self.nk)
                    .map(|kpt_idx| {
                        let simcc_x_slice = simcc_x_array.slice(s![batch_idx, kpt_idx, ..]);
                        let simcc_y_slice = simcc_y_array.slice(s![batch_idx, kpt_idx, ..]);
                        let (x_loc, max_val_x) = Self::argmax_and_max(&simcc_x_slice);
                        let (y_loc, max_val_y) = Self::argmax_and_max(&simcc_y_slice);
                        let confidence = 0.5 * (max_val_x + max_val_y);

                        if confidence > self.kconfs[kpt_idx] {
                            let x = x_loc as f32 * x_factor + x_offset;
                            let y = y_loc as f32 * y_factor + y_offset;

                            let mut kpt = Keypoint::from((x, y))
                                .with_confidence(confidence)
                                .with_id(kpt_idx);

                            if has_names {
                                kpt = kpt.with_name(&self.names[kpt_idx]);
                            }
                            kpt
                        } else {
                            Keypoint::default()
                        }
                    })
                    .collect();

                Y::default().with_keypointss(&[keypoints])
            })
            .collect();

        Ok(results)
    }

    fn argmax_and_max(arr: &ndarray::ArrayView1<f32>) -> (usize, f32) {
        let mut max_idx = 0;
        let mut max_val = arr[0];

        for (idx, &val) in arr.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }

        (max_idx, max_val)
    }

    fn hbb2cs(hbb: &Hbb, padding: f32) -> (Keypoint, Keypoint) {
        let (x1, y1, x2, y2) = hbb.xyxy();
        (
            ((x1 + x2) * 0.5, (y1 + y2) * 0.5).into(),
            ((x2 - x1) * padding, (y2 - y1) * padding).into(),
        )
    }

    fn get_warp_matrix(
        center: &Keypoint,
        scale: &Keypoint,
        rot: f32,
        output_size: (i32, i32),
        shift: (f32, f32),
        inv: bool,
    ) -> Vec<f32> {
        fn get_3rd_point(a: &Keypoint, b: &Keypoint) -> Keypoint {
            let direction = Keypoint::new(a.x() - b.x(), a.y() - b.y());
            (b.x() - direction.y(), b.y() + direction.x()).into()
        }

        let shift = Keypoint::new(shift.0, shift.1);
        let src_w = scale.x();
        let dst_w = output_size.0 as f32;
        let dst_h = output_size.1 as f32;
        let rot_rad = rot * PI / 180.0;
        let src_dir = Keypoint::new(0.0, src_w * -0.5).rotate(rot_rad);
        let dst_dir = Keypoint::new(0.0, dst_w * -0.5);
        let src_0 = Keypoint::new(
            center.x() + scale.x() * shift.x(),
            center.y() + scale.y() * shift.y(),
        );
        let src_1 = Keypoint::new(
            center.x() + src_dir.x() + scale.x() * shift.x(),
            center.y() + src_dir.y() + scale.y() * shift.y(),
        );
        let src_2 = get_3rd_point(&src_0, &src_1);
        let dst_0 = Keypoint::new(dst_w * 0.5, dst_h * 0.5);
        let dst_1 = Keypoint::new(dst_w * 0.5 + dst_dir.x(), dst_h * 0.5 + dst_dir.y());
        let dst_2 = get_3rd_point(&dst_0, &dst_1);
        let (src_points, dst_points) = if inv {
            (vec![&dst_0, &dst_1, &dst_2], vec![&src_0, &src_1, &src_2])
        } else {
            (vec![&src_0, &src_1, &src_2], vec![&dst_0, &dst_1, &dst_2])
        };

        let x1 = src_points[0].x();
        let y1 = src_points[0].y();
        let x2 = src_points[1].x();
        let y2 = src_points[1].y();
        let x3 = src_points[2].x();
        let y3 = src_points[2].y();

        let u1 = dst_points[0].x();
        let v1 = dst_points[0].y();
        let u2 = dst_points[1].x();
        let v2 = dst_points[1].y();
        let u3 = dst_points[2].x();
        let v3 = dst_points[2].y();

        let det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);

        if det.abs() < 1e-6 {
            return vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        }

        let m00 = (u1 * (y2 - y3) + u2 * (y3 - y1) + u3 * (y1 - y2)) / det;
        let m01 = (x1 * (u3 - u2) + x2 * (u1 - u3) + x3 * (u2 - u1)) / det;
        let m02 =
            (x1 * (y2 * u3 - y3 * u2) + x2 * (y3 * u1 - y1 * u3) + x3 * (y1 * u2 - y2 * u1)) / det;
        let m10 = (v1 * (y2 - y3) + v2 * (y3 - y1) + v3 * (y1 - y2)) / det;
        let m11 = (x1 * (v2 - v3) + x2 * (v3 - v1) + x3 * (v1 - v2)) / det;
        let m12 =
            (x1 * (y2 * v3 - y3 * v2) + x2 * (y3 * v1 - y1 * v3) + x3 * (y1 * v2 - y2 * v1)) / det;

        vec![m00, m01, m02, m10, m11, m12]
    }

    fn warp_affine(img: &Image, warp_mat: &[f32], output_size: (i32, i32)) -> Result<Image> {
        let (width, height) = output_size;
        let img_w = img.width();
        let img_h = img.height();

        let m00 = warp_mat[0];
        let m01 = warp_mat[1];
        let m02 = warp_mat[2];
        let m10 = warp_mat[3];
        let m11 = warp_mat[4];
        let m12 = warp_mat[5];

        let det = m00 * m11 - m01 * m10;
        if det.abs() < 1e-6 {
            return Image::from_u8s(
                &vec![0u8; (height * width * 3) as usize],
                width as u32,
                height as u32,
            );
        }

        let inv_det = 1.0 / det;
        let inv_m00 = m11 * inv_det;
        let inv_m01 = -m01 * inv_det;
        let inv_m10 = -m10 * inv_det;
        let inv_m11 = m00 * inv_det;
        let inv_m02 = (m01 * m12 - m11 * m02) * inv_det;
        let inv_m12 = (m10 * m02 - m00 * m12) * inv_det;
        let img_data = img.as_raw();

        let mut result_data = vec![0u8; (height * width * 3) as usize];
        result_data
            .par_chunks_exact_mut((width * 3) as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let dst_y = y as f32;
                for x in 0..width {
                    let dst_x = x as f32;
                    let src_x = inv_m00 * dst_x + inv_m01 * dst_y + inv_m02;
                    let src_y = inv_m10 * dst_x + inv_m11 * dst_y + inv_m12;

                    let x0 = src_x.floor() as i32;
                    let y0 = src_y.floor() as i32;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;

                    if x0 >= 0 && x1 < img_w as i32 && y0 >= 0 && y1 < img_h as i32 {
                        let dx = src_x - x0 as f32;
                        let dy = src_y - y0 as f32;
                        let w00 = (1.0 - dx) * (1.0 - dy);
                        let w01 = dx * (1.0 - dy);
                        let w10 = (1.0 - dx) * dy;
                        let w11 = dx * dy;

                        let dst_idx = (x * 3) as usize;
                        let y0_offset = (y0 as u32 * img_w) as usize * 3;
                        let y1_offset = (y1 as u32 * img_w) as usize * 3;
                        let x0_offset = x0 as usize * 3;
                        let x1_offset = x1 as usize * 3;

                        let src_idx0 = y0_offset + x0_offset;
                        let src_idx1 = y0_offset + x1_offset;
                        let src_idx2 = y1_offset + x0_offset;
                        let src_idx3 = y1_offset + x1_offset;

                        for c in 0..3 {
                            row[dst_idx + c] = (img_data[src_idx0 + c] as f32 * w00
                                + img_data[src_idx1 + c] as f32 * w01
                                + img_data[src_idx2 + c] as f32 * w10
                                + img_data[src_idx3 + c] as f32 * w11)
                                .round() as u8;
                        }
                    }
                }
            });

        Image::from_u8s(&result_data, width as u32, height as u32)
    }

    fn top_down_affine(
        input_size: (i32, i32),
        scale: &Keypoint,
        center: &Keypoint,
        img: &Image,
    ) -> Result<(Image, Keypoint)> {
        let (w, h) = input_size;
        let aspect_ratio = w as f32 / h as f32;
        let b_w = scale.x();
        let b_h = scale.y();
        let scale = if b_w > b_h * aspect_ratio {
            Keypoint::new(b_w, b_w / aspect_ratio)
        } else {
            Keypoint::new(b_h * aspect_ratio, b_h)
        };
        let rot = 0.0;
        let warp_mat = Self::get_warp_matrix(center, &scale, rot, (w, h), (0.0, 0.0), false);
        let img = Self::warp_affine(img, &warp_mat, (w, h))?;

        Ok((img, scale))
    }
}
