use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{
    inputs, Config, DynConf, Engine, Engines, FromConfig, Hbb, Image, ImageProcessor, Mask, Model,
    Module, Ops, ResizeModeType, Xs, X, Y,
};

/// ECDetSeg: EdgeCrafter real-time instance segmentation.
///
/// Detection front-end matches the RT-DETR / D-FINE family (inputs `images` +
/// `orig_target_sizes`, outputs `labels`, `boxes`, `scores`) with an additional
/// `masks` output decoded into per-instance masks like RF-DETR segmentation.
#[derive(Debug, Builder)]
pub struct ECDetSeg {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub names: Vec<String>,
    pub confs: DynConf,
    pub processor: ImageProcessor,
    pub spec: String,
    pub classes_excluded: Vec<usize>,
    pub classes_retained: Vec<usize>,
}

impl Model for ECDetSeg {
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
            engine.try_height().unwrap_or(&640.into()).opt(),
            engine.try_width().unwrap_or(&640.into()).opt(),
        );
        let spec = engine.spec().to_owned();
        let names: Vec<String> = config.inference.class_names;
        let confs = DynConf::new_or_default(&config.inference.class_confs, names.len());
        let classes_excluded = config.inference.classes_excluded;
        let classes_retained = config.inference.classes_retained;
        if !classes_excluded.is_empty() {
            tracing::info!("classes_excluded: {classes_excluded:?}");
        }
        if !classes_retained.is_empty() {
            tracing::info!("classes_retained: {classes_retained:?}");
        }
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            height,
            width,
            batch,
            spec,
            names,
            confs,
            processor,
            classes_excluded,
            classes_retained,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let x1 = crate::perf!("ECDetSeg::preprocess", self.processor.process(images)?);
        let x2 = X::from(vec![self.height as f32, self.width as f32])
            .insert_axis(0)?
            .repeat(0, self.batch)?;
        let ys = crate::perf!(
            "ECDetSeg::inference",
            engines.run(&Module::Model, inputs![&x1, x2]?)?
        );
        crate::perf!("ECDetSeg::postprocess", self.postprocess(&ys))
    }
}

impl ECDetSeg {
    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let resize_mode = match self.processor.resize_mode_type() {
            Some(ResizeModeType::Letterbox) => ResizeModeType::Letterbox,
            Some(ResizeModeType::FitAdaptive) => ResizeModeType::FitAdaptive,
            Some(ResizeModeType::FitExact) => ResizeModeType::FitExact,
            Some(x) => anyhow::bail!("Unsupported resize mode for ECDetSeg postprocess: {x:?}. Supported: FitExact, FitAdaptive, Letterbox"),
            _ => anyhow::bail!("No resize mode specified. Supported: FitExact, FitAdaptive, Letterbox"),
        };

        // Detection outputs match the RT-DETR / D-FINE export: labels (i64),
        // boxes (xyxy in model-input space), scores (sigmoid'd).
        let labels = outputs
            .get::<i64>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get labels"))?;
        let boxes = outputs
            .get::<f32>(1)
            .ok_or_else(|| anyhow::anyhow!("Failed to get bboxes"))?;
        let scores = outputs
            .get::<f32>(2)
            .ok_or_else(|| anyhow::anyhow!("Failed to get scores"))?;
        // Optional `masks` output: per-query mask logits at the prototype resolution.
        let preds_masks = outputs.get::<f32>(3);

        let ys: Vec<Y> = labels
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(boxes.axis_iter(Axis(0)).into_par_iter())
            .zip(scores.axis_iter(Axis(0)).into_par_iter())
            .enumerate()
            .filter_map(|(idx, ((labels, boxes), scores))| {
                let info = &self.processor.images_transform_info[idx];
                let (image_height, image_width) = (info.height_src, info.width_src);
                let dets: Vec<(Hbb, Option<Mask>)> = scores
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &score)| {
                        let class_id = labels[i] as usize;
                        if score < self.confs[class_id] {
                            return None;
                        }

                        if !self.classes_excluded.is_empty()
                            && self.classes_excluded.contains(&class_id)
                        {
                            return None;
                        }

                        if !self.classes_retained.is_empty()
                            && !self.classes_retained.contains(&class_id)
                        {
                            return None;
                        }

                        // Map xyxy box from model-input space back to the source image.
                        let xyxy_raw = boxes.slice(s![i, ..]);
                        let (x1, y1, x2, y2) = match resize_mode {
                            ResizeModeType::FitExact => {
                                let scale_x = image_width as f32 / self.width as f32;
                                let scale_y = image_height as f32 / self.height as f32;
                                (
                                    xyxy_raw[0] * scale_x,
                                    xyxy_raw[1] * scale_y,
                                    xyxy_raw[2] * scale_x,
                                    xyxy_raw[3] * scale_y,
                                )
                            }
                            ResizeModeType::Letterbox => {
                                let ratio = info.height_scale;
                                let pad_w = info.width_pad;
                                let pad_h = info.height_pad;
                                (
                                    (xyxy_raw[0] - pad_w) / ratio,
                                    (xyxy_raw[1] - pad_h) / ratio,
                                    (xyxy_raw[2] - pad_w) / ratio,
                                    (xyxy_raw[3] - pad_h) / ratio,
                                )
                            }
                            ResizeModeType::FitAdaptive => {
                                let ratio = info.height_scale;
                                (
                                    xyxy_raw[0] / ratio,
                                    xyxy_raw[1] / ratio,
                                    xyxy_raw[2] / ratio,
                                    xyxy_raw[3] / ratio,
                                )
                            }
                            _ => unreachable!(),
                        };

                        let x1 = x1.max(0.0).min(image_width as _);
                        let y1 = y1.max(0.0).min(image_height as _);
                        let x2 = x2.max(0.0).min(image_width as _);
                        let y2 = y2.max(0.0).min(image_height as _);
                        let mut hbb = Hbb::default()
                            .with_xyxy(x1, y1, x2, y2)
                            .with_confidence(score)
                            .with_id(class_id);
                        if !self.names.is_empty() {
                            hbb = hbb.with_name(&self.names[class_id]);
                        }

                        // Decode the per-instance mask (raw logits -> binarized at 0).
                        if let Some(preds_masks) = &preds_masks {
                            let mask = preds_masks.slice(s![idx, i, .., ..]);
                            let (mh, mw) = (mask.shape()[0], mask.shape()[1]);
                            let mask = mask.into_owned().into_raw_vec_and_offset().0;
                            let mask = Ops::resize_mask_with_mode(
                                mask,
                                mw,
                                mh,
                                image_width as _,
                                image_height as _,
                                self.width as _,
                                self.height as _,
                                resize_mode,
                                info,
                                crate::ResizeFilter::Bilinear,
                            )
                            .ok()?;

                            let mut mask =
                                Mask::new(&mask, image_width as _, image_height as _).ok()?;
                            mask = mask.with_id(class_id).with_confidence(score);
                            if !self.names.is_empty() {
                                mask = mask.with_name(&self.names[class_id]);
                            }
                            Some((hbb, Some(mask)))
                        } else {
                            Some((hbb, None))
                        }
                    })
                    .collect();

                let (y_hbbs, y_masks): (Vec<_>, Vec<_>) = dets.into_iter().unzip();

                Some(
                    Y::default()
                        .with_hbbs(&y_hbbs)
                        .with_masks(&y_masks.into_iter().flatten().collect::<Vec<_>>()),
                )
            })
            .collect();

        Ok(ys)
    }
}
