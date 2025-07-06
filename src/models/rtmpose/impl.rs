use aksr::Builder;
use anyhow::Result;
use ndarray::s;
use rayon::prelude::*;
use std::f32::consts::PI;

use crate::{elapsed_module, Config, DynConf, Engine, Hbb, Image, Keypoint, Processor, Xs, Y};

struct CentersAndScales {
    pub centers: Vec<Keypoint>,
    pub scales: Vec<Keypoint>,
}

#[derive(Builder, Debug)]
pub struct RTMPose {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,

    spec: String,
    processor: Processor,
    nk: usize,
    kconfs: DynConf,
    names: Vec<String>,
    simcc_split_ratio: f32,
}

impl RTMPose {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let spec = engine.spec().to_string();
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&256.into()).opt(),
            engine.try_width().unwrap_or(&192.into()).opt(),
        );
        let nk = config.nk().unwrap_or(17);
        let kconfs = DynConf::new_or_default(config.keypoint_confs(), nk);
        let names = config.keypoint_names().to_vec();
        let simcc_split_ratio = 2.0;
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            engine,
            height,
            width,
            batch,

            spec,
            processor,
            nk,
            kconfs,
            names,
            simcc_split_ratio,
        })
    }

    fn preprocess_one(
        img: &Image,
        hbb: Option<&Hbb>,
        model_input_size: (usize, usize),
    ) -> Result<(Image, CentersAndScales)> {
        let hbb = if let Some(hbb) = hbb {
            hbb
        } else {
            &Hbb::from_xyxy(0.0, 0.0, img.width() as f32, img.height() as f32)
        };
        let (center, scale) = Self::hbb2cs(hbb, 1.25);
        let (resized_img, scale) = Self::top_down_affine(
            (model_input_size.0 as i32, model_input_size.1 as i32),
            &scale,
            &center,
            img,
        )?;

        Ok((
            resized_img,
            CentersAndScales {
                centers: vec![center],
                scales: vec![scale],
            },
        ))
    }

    fn preprocess(&mut self, x: &Image, hbbs: Option<&[Hbb]>) -> Result<(Xs, CentersAndScales)> {
        let results: Result<Vec<_>> = match hbbs {
            None | Some(&[]) => vec![Self::preprocess_one(x, None, (self.width, self.height))]
                .into_iter()
                .collect(),
            Some(hbbs) => hbbs
                .par_iter()
                .map(|hbb| Self::preprocess_one(x, Some(hbb), (self.width, self.height)))
                .collect(),
        };
        let (processed_images, centers_and_scales): (Vec<_>, Vec<_>) = results?.into_iter().unzip();
        let mut centerss = Vec::new();
        let mut scaless = Vec::new();
        for cs in centers_and_scales {
            centerss.extend(cs.centers);
            scaless.extend(cs.scales);
        }
        // TODO
        let x = self.processor.process_images(&processed_images)?;

        // Update batch size
        self.batch = match hbbs {
            None | Some(&[]) => 1,
            Some(hbbs) => hbbs.len(),
        };

        Ok((
            x.into(),
            CentersAndScales {
                centers: centerss,
                scales: scaless,
            },
        ))
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, x: &Image, hbbs: Option<&[Hbb]>) -> Result<Y> {
        let (ys, centers_and_scales) =
            elapsed_module!("RTMPose", "preprocess", self.preprocess(x, hbbs)?);
        let ys = elapsed_module!("RTMPose", "inference", self.inference(ys)?);
        let y = elapsed_module!("RTMPose", "postprocess", {
            self.postprocess(ys, centers_and_scales)?
        });

        Ok(y)
    }

    fn postprocess(&mut self, xs: Xs, centers_and_scales: CentersAndScales) -> Result<Y> {
        // Update nk
        self.nk = xs[0].shape()[1];

        let simcc_x_array = &xs[0];
        let simcc_y_array = &xs[1];
        let y_kpts: Vec<Vec<Keypoint>> = (0..self.batch)
            .into_par_iter()
            .map(|batch_idx| {
                let mut keypoints = Vec::new();
                let center = &centers_and_scales.centers[batch_idx];
                let scale = &centers_and_scales.scales[batch_idx];
                for kpt_idx in 0..self.nk {
                    let simcc_x_slice = simcc_x_array.slice(s![batch_idx, kpt_idx, ..]);
                    let simcc_y_slice = simcc_y_array.slice(s![batch_idx, kpt_idx, ..]);
                    let (x_loc, max_val_x) = Self::argmax_and_max(&simcc_x_slice);
                    let (y_loc, max_val_y) = Self::argmax_and_max(&simcc_y_slice);
                    let confidence = 0.5 * (max_val_x + max_val_y);
                    let mut x = x_loc as f32 / self.simcc_split_ratio;
                    let mut y = y_loc as f32 / self.simcc_split_ratio;

                    // keypoints = keypoints / model_input_size * scale + center - scale / 2
                    let keypoint = if confidence > self.kconfs[kpt_idx] {
                        x = x / self.width as f32 * scale.x() + center.x() - scale.x() / 2.0;
                        y = y / self.height as f32 * scale.y() + center.y() - scale.y() / 2.0;

                        let mut kpt = Keypoint::from((x, y))
                            .with_confidence(confidence)
                            .with_id(kpt_idx);

                        if !self.names.is_empty() {
                            kpt = kpt.with_name(&self.names[kpt_idx]);
                        }
                        kpt
                    } else {
                        Keypoint::default()
                    };

                    keypoints.push(keypoint);
                }

                keypoints
            })
            .collect();

        Ok(Y::default().with_keypointss(&y_kpts))
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
        let mut result_data = vec![0u8; (height * width * 3) as usize];
        let img_rgb = img.to_rgb8();
        let (img_w, img_h) = img_rgb.dimensions();

        let m00 = warp_mat[0];
        let m01 = warp_mat[1];
        let m02 = warp_mat[2];
        let m10 = warp_mat[3];
        let m11 = warp_mat[4];
        let m12 = warp_mat[5];

        let det = m00 * m11 - m01 * m10;
        if det.abs() < 1e-6 {
            return Image::from_u8s(&result_data, width as u32, height as u32);
        }

        let inv_det = 1.0 / det;
        let inv_m00 = m11 * inv_det;
        let inv_m01 = -m01 * inv_det;
        let inv_m10 = -m10 * inv_det;
        let inv_m11 = m00 * inv_det;
        let inv_m02 = (m01 * m12 - m11 * m02) * inv_det;
        let inv_m12 = (m10 * m02 - m00 * m12) * inv_det;
        let img_data = img_rgb.as_raw();

        for y in 0..height {
            for x in 0..width {
                let dst_x = x as f32;
                let dst_y = y as f32;
                let src_x = inv_m00 * dst_x + inv_m01 * dst_y + inv_m02;
                let src_y = inv_m10 * dst_x + inv_m11 * dst_y + inv_m12;
                let x0 = src_x.floor() as i32;
                let y0 = src_y.floor() as i32;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                if x0 >= 0 && x1 < img_w as i32 && y0 >= 0 && y1 < img_h as i32 {
                    let dx = src_x - x0 as f32;
                    let dy = src_y - y0 as f32;
                    let dst_idx = ((y * width + x) * 3) as usize;

                    for c in 0..3 {
                        let src_idx_00 = ((y0 as u32 * img_w + x0 as u32) * 3 + c as u32) as usize;
                        let src_idx_01 = ((y0 as u32 * img_w + x1 as u32) * 3 + c as u32) as usize;
                        let src_idx_10 = ((y1 as u32 * img_w + x0 as u32) * 3 + c as u32) as usize;
                        let src_idx_11 = ((y1 as u32 * img_w + x1 as u32) * 3 + c as u32) as usize;
                        let p00 = img_data[src_idx_00] as f32;
                        let p01 = img_data[src_idx_01] as f32;
                        let p10 = img_data[src_idx_10] as f32;
                        let p11 = img_data[src_idx_11] as f32;
                        let interpolated = p00 * (1.0 - dx) * (1.0 - dy)
                            + p01 * dx * (1.0 - dy)
                            + p10 * (1.0 - dx) * dy
                            + p11 * dx * dy;

                        result_data[dst_idx + c] = interpolated.round() as u8;
                    }
                }
            }
        }

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
