use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rand::{prelude::*, rng};

use crate::{
    elapsed_module, inputs, Config, DynConf, Engine, Engines, FromConfig, Image, ImageProcessor,
    Mask, Model, Module, Ops, Polygon, SamKind, SamPrompt, X, Y,
};

/// Segment Anything Model (SAM) for image segmentation.
///
/// A foundation model for generating high-quality object masks from input prompts such as points or boxes.
/// Supports multiple variants including the original SAM, SAM2, MobileSAM, SAM-HQ and EdgeSAM.
#[derive(Builder, Debug)]
pub struct SAM {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub processor: ImageProcessor,
    pub conf: DynConf,
    pub find_contours: bool,
    pub kind: SamKind,
    pub use_low_res_mask: bool,
    pub spec: String,
}

impl Model for SAM {
    type Input<'a> = (&'a [Image], &'a [SamPrompt]);

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let encoder = Engine::from_config(config.take_module(&Module::Encoder)?)?;
        let decoder = Engine::from_config(config.take_module(&Module::Decoder)?)?;

        let (batch, height, width) = (
            encoder.batch().opt(),
            encoder.try_height().unwrap_or(&1024.into()).opt(),
            encoder.try_width().unwrap_or(&1024.into()).opt(),
        );

        let spec = encoder.spec().to_owned();

        let conf = DynConf::new_or_default(&config.inference.class_confs, 1);
        let find_contours = config.inference.find_contours;
        let kind = match config.inference.sam_kind {
            Some(x) => x,
            None => anyhow::bail!("Error: no clear `SamKind` specified."),
        };
        let use_low_res_mask = match kind {
            SamKind::Sam | SamKind::MobileSam | SamKind::SamHq => {
                config.inference.sam_low_res_mask.unwrap_or(false)
            }
            SamKind::EdgeSam | SamKind::Sam2 => true,
        };

        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            conf,
            batch,
            height,
            width,
            processor,
            kind,
            find_contours,
            use_low_res_mask,
            spec,
        };

        let mut engines = Engines::default();
        engines.insert(Module::Encoder, encoder);
        engines.insert(Module::Decoder, decoder);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, (images, prompts): Self::Input<'_>) -> Result<Vec<Y>> {
        let embeddings = elapsed_module!("SAM", "encode", self.encode(engines, images)?);
        elapsed_module!("SAM", "decode", self.decode(engines, &embeddings, prompts))
    }
}

impl SAM {
    fn encode(&mut self, engines: &mut Engines, xs: &[Image]) -> Result<Vec<X>> {
        let xs_ = self.processor.process(xs)?;
        let output = engines.run(&Module::Encoder, inputs![&xs_]?)?;
        let xs_out: Vec<X> = (0..output.len())
            .map(|i| X::from(output.get::<f32>(i).unwrap()))
            .collect();
        Ok(xs_out)
    }

    fn decode(&mut self, engines: &mut Engines, xs: &[X], prompts: &[SamPrompt]) -> Result<Vec<Y>> {
        let (image_embeddings, high_res_features_0, high_res_features_1) = match self.kind {
            SamKind::Sam2 => (&xs[0], Some(&xs[1]), Some(&xs[2])),
            _ => (&xs[0], None, None),
        };

        let mut ys: Vec<Y> = Vec::new();
        for (idx, image_embedding) in image_embeddings.axis_iter(Axis(0)).enumerate() {
            let info = &self.processor.images_transform_info[idx];
            let (image_height, image_width, ratio) =
                (info.height_src, info.width_src, info.height_scale);

            let (mut point_coords, mut point_labels) = (
                prompts[idx].point_coords(ratio)?,
                prompts[idx].point_labels()?,
            );

            if point_coords.shape()[0] != 1 {
                point_coords = X::from(point_coords.slice(s![-1, .., ..]).to_owned().into_dyn())
                    .insert_axis(0)?;
            }
            if point_labels.shape()[0] != 1 {
                point_labels = X::from(point_labels.slice(s![-1, ..,]).to_owned().into_dyn())
                    .insert_axis(0)?;
            }

            let args = match self.kind {
                SamKind::Sam | SamKind::MobileSam => {
                    vec![
                        X::from(image_embedding.into_dyn().into_owned())
                            .insert_axis(0)?
                            .repeat(0, self.batch)?, // image_embedding
                        point_coords,
                        point_labels,
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
                        point_coords,
                        point_labels,
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
                        point_coords,
                        point_labels,
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
                        point_coords,
                        point_labels,
                        X::zeros(&[1, 1, self.height_low_res() as _, self.width_low_res() as _]),
                        X::zeros(&[1]),
                        X::from(vec![image_height as _, image_width as _]),
                    ]
                }
            };

            let ys_ = engines.run(&Module::Decoder, &args)?;

            let mut y_masks: Vec<Mask> = Vec::new();
            let mut y_polygons: Vec<Polygon> = Vec::new();

            // masks & confs - extract as owned X to avoid lifetime issues
            let (masks, confs) =
                match self.kind {
                    SamKind::Sam | SamKind::MobileSam | SamKind::SamHq => {
                        if !self.use_low_res_mask {
                            (
                                X::from(
                                    ys_.get::<f32>(0)
                                        .ok_or_else(|| anyhow::anyhow!("Failed to get masks"))?,
                                ),
                                X::from(
                                    ys_.get::<f32>(1)
                                        .ok_or_else(|| anyhow::anyhow!("Failed to get confs"))?,
                                ),
                            )
                        } else {
                            (
                                X::from(ys_.get::<f32>(2).ok_or_else(|| {
                                    anyhow::anyhow!("Failed to get low-res masks")
                                })?),
                                X::from(
                                    ys_.get::<f32>(1)
                                        .ok_or_else(|| anyhow::anyhow!("Failed to get confs"))?,
                                ),
                            )
                        }
                    }
                    SamKind::Sam2 => (
                        X::from(
                            ys_.get::<f32>(0)
                                .ok_or_else(|| anyhow::anyhow!("Failed to get masks"))?,
                        ),
                        X::from(
                            ys_.get::<f32>(1)
                                .ok_or_else(|| anyhow::anyhow!("Failed to get confs"))?,
                        ),
                    ),
                    SamKind::EdgeSam => (
                        X::from(
                            ys_.get_by_name::<f32>("masks")
                                .ok_or_else(|| anyhow::anyhow!("Failed to get masks"))?,
                        ),
                        X::from(
                            ys_.get_by_name::<f32>("scores")
                                .ok_or_else(|| anyhow::anyhow!("Failed to get scores"))?,
                        ),
                    ),
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

                // contours
                let mut rng = rng();
                let id = rng.random_range(0..20);
                let mask = Mask::new(&luma, image_width, image_height)?.with_id(id);
                if self.find_contours {
                    for polygon in mask.polygons().into_iter() {
                        y_polygons.push(polygon.with_confidence(iou[0]).with_id(id));
                    }
                }
                y_masks.push(mask);
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

    /// Returns the width of the low-resolution feature maps.
    pub fn width_low_res(&self) -> usize {
        self.width / 4
    }

    /// Returns the height of the low-resolution feature maps.  
    pub fn height_low_res(&self) -> usize {
        self.height / 4
    }
}
