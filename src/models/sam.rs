use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Axis};
use rand::prelude::*;

use crate::{DynConf, Mask, MinOptMax, Ops, Options, OrtEngine, Polygon, Xs, X, Y};

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum SamKind {
    Sam,
    Sam2,
    MobileSam,
    SamHq,
    EdgeSam,
}

#[derive(Debug, Default, Clone)]
pub struct SamPrompt {
    points: Vec<f32>,
    labels: Vec<f32>,
}

impl SamPrompt {
    pub fn everything() -> Self {
        todo!()
    }

    pub fn with_postive_point(mut self, x: f32, y: f32) -> Self {
        self.points.extend_from_slice(&[x, y]);
        self.labels.push(1.);
        self
    }

    pub fn with_negative_point(mut self, x: f32, y: f32) -> Self {
        self.points.extend_from_slice(&[x, y]);
        self.labels.push(0.);
        self
    }

    pub fn with_bbox(mut self, x: f32, y: f32, x2: f32, y2: f32) -> Self {
        self.points.extend_from_slice(&[x, y, x2, y2]);
        self.labels.extend_from_slice(&[2., 3.]);
        self
    }

    pub fn point_coords(&self, r: f32) -> Result<X> {
        let point_coords = Array::from_shape_vec((1, self.num_points(), 2), self.points.clone())?
            .into_dyn()
            .into_owned();
        Ok(X::from(point_coords * r))
    }

    pub fn point_labels(&self) -> Result<X> {
        let point_labels = Array::from_shape_vec((1, self.num_points()), self.labels.clone())?
            .into_dyn()
            .into_owned();
        Ok(X::from(point_labels))
    }

    pub fn num_points(&self) -> usize {
        self.points.len() / 2
    }
}

#[derive(Debug)]
pub struct SAM {
    encoder: OrtEngine,
    decoder: OrtEngine,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    pub conf: DynConf,
    find_contours: bool,
    kind: SamKind,
    use_low_res_mask: bool,
}

impl SAM {
    pub fn new(options_encoder: Options, options_decoder: Options) -> Result<Self> {
        let mut encoder = OrtEngine::new(&options_encoder)?;
        let mut decoder = OrtEngine::new(&options_decoder)?;
        let (batch, height, width) = (
            encoder.inputs_minoptmax()[0][0].to_owned(),
            encoder.inputs_minoptmax()[0][2].to_owned(),
            encoder.inputs_minoptmax()[0][3].to_owned(),
        );
        let conf = DynConf::new(&options_decoder.confs, 1);

        let kind = match options_decoder.sam_kind {
            Some(x) => x,
            None => anyhow::bail!("Error: no clear `SamKind` specified."),
        };
        let find_contours = options_decoder.find_contours;
        let use_low_res_mask = match kind {
            SamKind::Sam | SamKind::MobileSam | SamKind::SamHq => {
                options_decoder.use_low_res_mask.unwrap_or(false)
            }
            SamKind::EdgeSam | SamKind::Sam2 => true,
        };

        encoder.dry_run()?;
        decoder.dry_run()?;

        Ok(Self {
            encoder,
            decoder,
            batch,
            height,
            width,
            conf,
            kind,
            find_contours,
            use_low_res_mask,
        })
    }

    pub fn run(&mut self, xs: &[DynamicImage], prompts: &[SamPrompt]) -> Result<Vec<Y>> {
        let ys = self.encode(xs)?;
        self.decode(&ys, xs, prompts)
    }

    pub fn encode(&mut self, xs: &[DynamicImage]) -> Result<Xs> {
        let xs_ = X::apply(&[
            Ops::Letterbox(
                xs,
                self.height() as u32,
                self.width() as u32,
                "Bilinear",
                0,
                "auto",
                false,
            ),
            Ops::Standardize(&[123.675, 116.28, 103.53], &[58.395, 57.12, 57.375], 3),
            Ops::Nhwc2nchw,
        ])?;

        self.encoder.run(Xs::from(xs_))
    }

    pub fn decode(
        &mut self,
        xs: &Xs,
        xs0: &[DynamicImage],
        prompts: &[SamPrompt],
    ) -> Result<Vec<Y>> {
        let (image_embeddings, high_res_features_0, high_res_features_1) = match self.kind {
            SamKind::Sam2 => (&xs[0], Some(&xs[1]), Some(&xs[2])),
            _ => (&xs[0], None, None),
        };

        let mut ys: Vec<Y> = Vec::new();
        for (idx, image_embedding) in image_embeddings.axis_iter(Axis(0)).enumerate() {
            let image_width = xs0[idx].width() as f32;
            let image_height = xs0[idx].height() as f32;
            let ratio =
                (self.width() as f32 / image_width).min(self.height() as f32 / image_height);
            let args = match self.kind {
                SamKind::Sam | SamKind::MobileSam => {
                    vec![
                        X::from(image_embedding.into_dyn().into_owned())
                            .insert_axis(0)?
                            .repeat(0, self.batch() as usize)?, // image_embedding
                        prompts[idx].point_coords(ratio)?, // point_coords
                        prompts[idx].point_labels()?,      // point_labels
                        X::zeros(&[1, 1, self.height_low_res() as _, self.width_low_res() as _]), // mask_input,
                        X::zeros(&[1]),                           // has_mask_input
                        X::from(vec![image_height, image_width]), // orig_im_size
                    ]
                }
                SamKind::SamHq => {
                    vec![
                        X::from(image_embedding.into_dyn().into_owned())
                            .insert_axis(0)?
                            .repeat(0, self.batch() as usize)?, // image_embedding
                        X::from(xs[1].slice(s![idx, .., .., ..]).into_dyn().into_owned())
                            .insert_axis(0)?
                            .insert_axis(0)?
                            .repeat(0, self.batch() as usize)?, // intern_embedding
                        prompts[idx].point_coords(ratio)?, // point_coords
                        prompts[idx].point_labels()?,      // point_labels
                        X::zeros(&[1, 1, self.height_low_res() as _, self.width_low_res() as _]), // mask_input
                        X::zeros(&[1]),                           // has_mask_input
                        X::from(vec![image_height, image_width]), // orig_im_size
                    ]
                }
                SamKind::EdgeSam => {
                    vec![
                        X::from(image_embedding.into_dyn().into_owned())
                            .insert_axis(0)?
                            .repeat(0, self.batch() as usize)?,
                        prompts[idx].point_coords(ratio)?,
                        prompts[idx].point_labels()?,
                    ]
                }
                SamKind::Sam2 => {
                    vec![
                        X::from(image_embedding.into_dyn().into_owned())
                            .insert_axis(0)?
                            .repeat(0, self.batch() as usize)?,
                        X::from(
                            high_res_features_0
                                .unwrap()
                                .slice(s![idx, .., .., ..])
                                .into_dyn()
                                .into_owned(),
                        )
                        .insert_axis(0)?
                        .repeat(0, self.batch() as usize)?,
                        X::from(
                            high_res_features_1
                                .unwrap()
                                .slice(s![idx, .., .., ..])
                                .into_dyn()
                                .into_owned(),
                        )
                        .insert_axis(0)?
                        .repeat(0, self.batch() as usize)?,
                        prompts[idx].point_coords(ratio)?,
                        prompts[idx].point_labels()?,
                        X::zeros(&[1, 1, self.height_low_res() as _, self.width_low_res() as _]), // mask_input
                        X::zeros(&[1]),                           // has_mask_input
                        X::from(vec![image_height, image_width]), // orig_im_size
                    ]
                }
            };

            let ys_ = self.decoder.run(Xs::from(args))?;

            let mut y_masks: Vec<Mask> = Vec::new();
            let mut y_polygons: Vec<Polygon> = Vec::new();

            // masks & confs
            let (masks, confs) = match self.kind {
                SamKind::Sam | SamKind::MobileSam | SamKind::SamHq => {
                    if !self.use_low_res_mask {
                        (&ys_[0], &ys_[1])
                        // (&ys_["masks"], &ys_["iou_predictions"])
                    } else {
                        (&ys_[2], &ys_[1])
                        // (&ys_["low_res_masks"], &ys_["iou_predictions"])
                    }
                }
                SamKind::Sam2 => (&ys_[0], &ys_[1]),
                SamKind::EdgeSam => (&ys_["masks"], &ys_["scores"]),
            };

            for (mask, iou) in masks.axis_iter(Axis(0)).zip(confs.axis_iter(Axis(0))) {
                let (i, conf) = match iou
                    .to_owned()
                    .into_raw_vec()
                    .into_iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                {
                    Some((i, c)) => (i, c),
                    None => continue,
                };

                if conf < self.conf[0] {
                    continue;
                }
                let mask = mask.slice(s![i, .., ..]);

                let (h, w) = mask.dim();
                let luma = if self.use_low_res_mask {
                    Ops::resize_lumaf32_vec(
                        &mask.to_owned().into_raw_vec(),
                        w as _,
                        h as _,
                        image_width as _,
                        image_height as _,
                        true,
                        "Bilinear",
                    )?
                } else {
                    mask.mapv(|x| if x > 0. { 255u8 } else { 0u8 })
                        .into_raw_vec()
                };

                let luma: image::ImageBuffer<image::Luma<_>, Vec<_>> =
                    match image::ImageBuffer::from_raw(image_width as _, image_height as _, luma) {
                        None => continue,
                        Some(x) => x,
                    };

                // contours
                let mut rng = thread_rng();
                let id = rng.gen_range(0..20);
                if self.find_contours {
                    let contours: Vec<imageproc::contours::Contour<i32>> =
                        imageproc::contours::find_contours_with_threshold(&luma, 0);
                    for c in contours.iter() {
                        let polygon = Polygon::default().with_points_imageproc(&c.points);
                        y_polygons.push(polygon.with_confidence(iou[0]).with_id(id));
                    }
                }
                y_masks.push(Mask::default().with_mask(luma).with_id(id));
            }

            let mut y = Y::default();
            if !y_masks.is_empty() {
                y = y.with_masks(&y_masks);
            }
            if !y_polygons.is_empty() {
                y = y.with_polygons(&y_polygons);
            }

            ys.push(y);
        }

        Ok(ys)
    }

    pub fn width_low_res(&self) -> usize {
        self.width() as usize / 4
    }

    pub fn height_low_res(&self) -> usize {
        self.height() as usize / 4
    }

    pub fn batch(&self) -> isize {
        self.batch.opt
    }

    pub fn width(&self) -> isize {
        self.width.opt
    }

    pub fn height(&self) -> isize {
        self.height.opt
    }
}
