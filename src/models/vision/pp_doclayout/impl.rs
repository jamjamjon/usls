use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{
    inputs, Config, DynConf, Engine, Engines, FromConfig, Hbb, Image, ImageProcessor, Mask, Model,
    Module, Ops, Version, Xs, X, Y,
};

/// PP-DocLayoutsss-v1/v2/v3
#[derive(Debug, Builder)]
pub struct PPDocLayout {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub max_det: usize,
    pub names: Vec<String>,
    pub confs: DynConf,
    pub processor: ImageProcessor,
    pub spec: String,
    pub version: Version,
}

impl Model for PPDocLayout {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&800.into()).opt(),
            engine.try_width().unwrap_or(&800.into()).opt(),
        );
        let spec = engine.spec().to_owned();
        let names: Vec<String> = config.inference.class_names;
        let confs = DynConf::new_or_default(&config.inference.class_confs, names.len());
        let version = config.version.unwrap_or_default();
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            height,
            width,
            batch,
            max_det: 300,
            spec,
            names,
            confs,
            processor,
            version,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let x0 = X::from(vec![self.height as f32, self.width as f32])
            .insert_axis(0)?
            .repeat(0, self.batch)?;
        let x1 = crate::perf!("PPDocLayout::preprocess", self.processor.process(images)?);
        let x2 = X::from_shape_vec_generic(
            [self.batch, 2],
            self.processor
                .images_transform_info
                .iter()
                .flat_map(|x| [x.height_scale, x.width_scale])
                .collect::<Vec<_>>(),
        )?;
        let ys = crate::perf!(
            "PPDocLayout::inference",
            engines.run(&Module::Model, inputs![x0, &x1, x2]?)?
        );
        crate::perf!("PPDocLayout::postprocess", self.postprocess(&ys))
    }
}

impl PPDocLayout {
    fn postprocess(&self, xs: &Xs) -> Result<Vec<Y>> {
        let preds = xs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get preds"))?;

        let f = |pred_slice: &[f32]| -> Option<Hbb> {
            let class_id = pred_slice[0] as usize;
            let score = pred_slice[1];

            if score < self.confs[class_id] {
                return None;
            }

            let mut hbb = Hbb::default()
                .with_xyxy(pred_slice[2], pred_slice[3], pred_slice[4], pred_slice[5])
                .with_confidence(score)
                .with_id(class_id);

            if !self.names.is_empty() {
                hbb = hbb.with_name(&self.names[class_id]);
            }
            Some(hbb)
        };

        match self.version {
            Version(1, _, _) => {
                // preds: [batch * max_det, 6]，[label_index, score, xmin, ymin, xmax, ymax]
                let preds_reshaped = preds.to_shape([self.batch, self.max_det, 6])?;
                let ys: Vec<Y> = preds_reshaped
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .filter_map(|preds| {
                        let y_bboxes: Vec<Hbb> = preds
                            .outer_iter()
                            .filter_map(|pred| pred.as_slice().and_then(f))
                            .collect();

                        Some(Y::default().with_hbbs(&y_bboxes))
                    })
                    .collect();

                Ok(ys)
            }
            Version(2, _, _) => {
                // preds: [batch * max_det, 8]，[label_index, score, xmin, ymin, xmax, ymax，row, col]
                let preds_reshaped = preds.to_shape([self.batch, self.max_det, 8])?;
                let ys: Vec<Y> = preds_reshaped
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .filter_map(|preds| {
                        let mut y_bboxes: Vec<(Hbb, usize, usize)> = preds
                            .outer_iter()
                            .filter_map(|pred| {
                                pred.as_slice().and_then(|slice| {
                                    f(slice).map(|hbb| (hbb, slice[6] as usize, slice[7] as usize))
                                })
                            })
                            .collect();

                        // Sort by reading order: first by row, then by column
                        y_bboxes.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2)));

                        let y_bboxes: Vec<Hbb> =
                            y_bboxes.into_iter().map(|(hbb, _, _)| hbb).collect();

                        Some(Y::default().with_hbbs(&y_bboxes))
                    })
                    .collect();

                Ok(ys)
            }
            Version(3, _, _) => {
                // preds: [batch * max_det, 7]，[label_index, score, xmin, ymin, xmax, ymax, reading_order]
                let preds_reshaped = preds.to_shape([self.batch, self.max_det, 7])?;
                let preds_masks = xs
                    .get::<f32>(2)
                    .ok_or_else(|| anyhow::anyhow!("Failed to get masks"))?;
                let preds_masks = preds_masks.to_shape([self.batch, self.max_det, 200, 200])?;

                let ys: Vec<Y> = preds_reshaped
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .zip(preds_masks.axis_iter(Axis(0)).into_par_iter())
                    .enumerate()
                    .filter_map(|(idx, (preds, preds_masks))| {
                        let info = &self.processor.images_transform_info[idx];
                        let (image_height, image_width) = (info.height_src, info.width_src);

                        let mut items: Vec<(Hbb, Mask, usize)> = preds
                            .outer_iter()
                            .zip(preds_masks.outer_iter())
                            .filter_map(|(pred_slice, pred_mask)| {
                                let slice = pred_slice.as_slice()?;
                                let hbb = f(slice)?;
                                let order = slice[6] as usize;
                                let (mh, mw) = (pred_mask.shape()[0], pred_mask.shape()[1]);
                                let (ih, iw) = (image_height, image_width);
                                let mask_f32: Vec<f32> =
                                    pred_mask.iter().map(|&x| 1. / (1. + (-x).exp())).collect();
                                let mask_f32: Vec<f32> = Ops::interpolate_1d(
                                    &mask_f32, mw as _, mh as _, iw as _, ih as _, false,
                                )
                                .ok()?;
                                let mask_u8: Vec<u8> = mask_f32
                                    .into_iter()
                                    .map(|x| if x <= 0.5 { 0 } else { 1 })
                                    .collect();

                                let mut mask = Mask::new(&mask_u8, iw as _, ih as _).ok()?;

                                if let Some(id) = hbb.id() {
                                    mask = mask.with_id(id);
                                }
                                if let Some(name) = hbb.name() {
                                    mask = mask.with_name(name);
                                }
                                if let Some(confidence) = hbb.confidence() {
                                    mask = mask.with_confidence(confidence);
                                }

                                Some((hbb, mask, order))
                            })
                            .collect();

                        items.sort_by(|a, b| a.2.cmp(&b.2));
                        let hbbs: Vec<Hbb> = items.iter().map(|(h, _, _)| h.clone()).collect();
                        let masks: Vec<Mask> = items.into_iter().map(|(_, m, _)| m).collect();
                        Some(Y::default().with_hbbs(&hbbs).with_masks(&masks))
                    })
                    .collect();

                Ok(ys)
            }
            _ => anyhow::bail!("Unsupported version"),
        }
    }
}
