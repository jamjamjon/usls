use aksr::Builder;
use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Axis};
use rand::prelude::*;

use crate::{elapsed, DynConf, Engine, Mask, Ops, Options, Polygon, Processor, Ts, Xs, Ys, X, Y};

#[derive(Debug, Clone)]
pub enum SamKind {
    Sam,
    Sam2,
    MobileSam,
    SamHq,
    EdgeSam,
}

impl TryFrom<&str> for SamKind {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "sam" => Ok(Self::Sam),
            "sam2" => Ok(Self::Sam2),
            "mobilesam" | "mobile-sam" => Ok(Self::MobileSam),
            "samhq" | "sam-hq" => Ok(Self::SamHq),
            "edgesam" | "edge-sam" => Ok(Self::EdgeSam),
            x => anyhow::bail!("Unsupported SamKind: {}", x),
        }
    }
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

#[derive(Builder, Debug)]
pub struct SAM {
    encoder: Engine,
    decoder: Engine,
    height: usize,
    width: usize,
    batch: usize,
    processor: Processor,
    conf: DynConf,
    find_contours: bool,
    kind: SamKind,
    use_low_res_mask: bool,
    ts: Ts,
    spec: String,
}

impl SAM {
    pub fn new(options_encoder: Options, options_decoder: Options) -> Result<Self> {
        let encoder = options_encoder.to_engine()?;
        let decoder = options_decoder.to_engine()?;
        let (batch, height, width) = (
            encoder.batch().opt(),
            encoder.try_height().unwrap_or(&1024.into()).opt(),
            encoder.try_width().unwrap_or(&1024.into()).opt(),
        );
        let ts = Ts::merge(&[encoder.ts(), decoder.ts()]);
        let spec = encoder.spec().to_owned();

        let processor = options_encoder
            .to_processor()?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let conf = DynConf::new(options_encoder.class_confs(), 1);
        let find_contours = options_encoder.find_contours;
        let kind = match options_encoder.sam_kind {
            Some(x) => x,
            None => anyhow::bail!("Error: no clear `SamKind` specified."),
        };
        let use_low_res_mask = match kind {
            SamKind::Sam | SamKind::MobileSam | SamKind::SamHq => {
                options_encoder.low_res_mask.unwrap_or(false)
            }
            SamKind::EdgeSam | SamKind::Sam2 => true,
        };

        Ok(Self {
            encoder,
            decoder,
            conf,
            batch,
            height,
            width,
            ts,
            processor,
            kind,
            find_contours,
            use_low_res_mask,
            spec,
        })
    }

    pub fn forward(&mut self, xs: &[DynamicImage], prompts: &[SamPrompt]) -> Result<Ys> {
        let ys = elapsed!("encode", self.ts, { self.encode(xs)? });
        let ys = elapsed!("decode", self.ts, { self.decode(&ys, prompts)? });

        Ok(ys)
    }

    pub fn encode(&mut self, xs: &[DynamicImage]) -> Result<Xs> {
        let xs_ = self.processor.process_images(xs)?;
        self.encoder.run(Xs::from(xs_))
    }

    pub fn decode(&mut self, xs: &Xs, prompts: &[SamPrompt]) -> Result<Ys> {
        let (image_embeddings, high_res_features_0, high_res_features_1) = match self.kind {
            SamKind::Sam2 => (&xs[0], Some(&xs[1]), Some(&xs[2])),
            _ => (&xs[0], None, None),
        };

        let mut ys: Vec<Y> = Vec::new();
        for (idx, image_embedding) in image_embeddings.axis_iter(Axis(0)).enumerate() {
            let (image_height, image_width) = self.processor.image0s_size[idx];
            let ratio = self.processor.scale_factors_hw[idx][0];

            let args = match self.kind {
                SamKind::Sam | SamKind::MobileSam => {
                    vec![
                        X::from(image_embedding.into_dyn().into_owned())
                            .insert_axis(0)?
                            .repeat(0, self.batch)?, // image_embedding
                        prompts[idx].point_coords(ratio)?, // point_coords
                        prompts[idx].point_labels()?,      // point_labels
                        X::zeros(&[1, 1, self.height_low_res() as _, self.width_low_res() as _]), // mask_input,
                        X::zeros(&[1]), // has_mask_input
                        X::from(vec![image_height as _, image_width as _]), // orig_im_size
                    ]
                }
                SamKind::SamHq => {
                    vec![
                        X::from(image_embedding.into_dyn().into_owned())
                            .insert_axis(0)?
                            .repeat(0, self.batch)?, // image_embedding
                        X::from(xs[1].slice(s![idx, .., .., ..]).into_dyn().into_owned())
                            .insert_axis(0)?
                            .insert_axis(0)?
                            .repeat(0, self.batch)?, // intern_embedding
                        prompts[idx].point_coords(ratio)?, // point_coords
                        prompts[idx].point_labels()?,      // point_labels
                        X::zeros(&[1, 1, self.height_low_res() as _, self.width_low_res() as _]), // mask_input
                        X::zeros(&[1]), // has_mask_input
                        X::from(vec![image_height as _, image_width as _]), // orig_im_size
                    ]
                }
                SamKind::EdgeSam => {
                    vec![
                        X::from(image_embedding.into_dyn().into_owned())
                            .insert_axis(0)?
                            .repeat(0, self.batch)?,
                        prompts[idx].point_coords(ratio)?,
                        prompts[idx].point_labels()?,
                    ]
                }
                SamKind::Sam2 => {
                    vec![
                        X::from(image_embedding.into_dyn().into_owned())
                            .insert_axis(0)?
                            .repeat(0, self.batch)?,
                        X::from(
                            high_res_features_0
                                .unwrap()
                                .slice(s![idx, .., .., ..])
                                .into_dyn()
                                .into_owned(),
                        )
                        .insert_axis(0)?
                        .repeat(0, self.batch)?,
                        X::from(
                            high_res_features_1
                                .unwrap()
                                .slice(s![idx, .., .., ..])
                                .into_dyn()
                                .into_owned(),
                        )
                        .insert_axis(0)?
                        .repeat(0, self.batch)?,
                        prompts[idx].point_coords(ratio)?,
                        prompts[idx].point_labels()?,
                        X::zeros(&[1, 1, self.height_low_res() as _, self.width_low_res() as _]), // mask_input
                        X::zeros(&[1]), // has_mask_input
                        X::from(vec![image_height as _, image_width as _]), // orig_im_size
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
                    .into_raw_vec_and_offset()
                    .0
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
                    Ops::resize_lumaf32_u8(
                        &mask.into_owned().into_raw_vec_and_offset().0,
                        w as _,
                        h as _,
                        image_width as _,
                        image_height as _,
                        true,
                        "Bilinear",
                    )?
                } else {
                    mask.mapv(|x| if x > 0. { 255u8 } else { 0u8 })
                        .into_raw_vec_and_offset()
                        .0
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
                        let polygon = Polygon::default().with_points_imageproc(&c.points).verify();
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

        Ok(ys.into())
    }

    pub fn width_low_res(&self) -> usize {
        self.width / 4
    }

    pub fn height_low_res(&self) -> usize {
        self.height / 4
    }
}
