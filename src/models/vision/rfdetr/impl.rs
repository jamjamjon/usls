use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{
    elapsed_module, ort_inputs, Config, DynConf, Engine, FromConfig, Hbb, Image, ImageProcessor,
    Mask, Module, Ops, Task, Y,
};

#[derive(Debug, Builder)]
pub struct RFDETR {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    names: Vec<String>,
    confs: DynConf,
    processor: ImageProcessor,
    spec: String,
    task: Task,
}

impl RFDETR {
    pub fn new(mut config: Config) -> Result<Self> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&560.into()).opt(),
            engine.try_width().unwrap_or(&560.into()).opt(),
        );
        let task = config.task.unwrap_or(Task::ObjectDetection);
        let spec = engine.spec().to_string();
        let names: Vec<String> = config.inference.class_names;
        let confs = DynConf::new_or_default(&config.inference.class_confs, names.len());
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            engine,
            height,
            width,
            batch,
            spec,
            names,
            confs,
            processor,
            task,
        })
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        // Preprocess
        let processed = elapsed_module!("RFDETR", "preprocess", self.processor.process(xs)?);

        // Inference and postprocess (combined due to lifetime constraints)
        let (has_masks, results) = {
            let ys = elapsed_module!("RFDETR", "inference", {
                self.engine.run(ort_inputs![&processed]?)?
            });

            let has_masks = ys.len() > 2;

            // Zero-copy access to outputs via Ys wrapper
            let bboxes = ys.get::<f32>(0)?;
            let logits = ys.get::<f32>(1)?;
            let masks = ys.try_get::<f32>(2);

            let results = elapsed_module!(
                "RFDETR",
                "postprocess",
                Self::postprocess_impl(
                    self.processor.images_transform_info(),
                    &self.confs,
                    &self.names,
                    self.width,
                    self.height,
                    &bboxes,
                    &logits,
                    masks.as_ref(),
                )?
            );

            (has_masks, results)
        };

        // Update task if masks present
        if has_masks {
            self.task = Task::InstanceSegmentation;
        }

        Ok(results)
    }

    #[allow(clippy::too_many_arguments)]
    fn postprocess_impl(
        images_transform_info: &[crate::ImageTransformInfo],
        confs: &DynConf,
        names: &[String],
        width: usize,
        height: usize,
        bboxes: &crate::XView<'_, f32>,
        logits: &crate::XView<'_, f32>,
        masks: Option<&crate::XView<'_, f32>>,
    ) -> Result<Vec<Y>> {
        let ys: Vec<Y> = logits
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, logits_batch)| {
                let (image_height, image_width) = (
                    images_transform_info[idx].height_src,
                    images_transform_info[idx].width_src,
                );
                let ratio = images_transform_info[idx].height_scale;
                let y_bboxes_masks: Vec<(Hbb, Option<Mask>)> = logits_batch
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .filter_map(|(i, clss)| {
                        let mut best_class_id: Option<usize> = None;
                        let mut best_conf: f32 = f32::NEG_INFINITY;
                        for (class_id, &logit) in clss.iter().enumerate() {
                            let conf = 1.0 / ((-logit).exp() + 1.0);
                            if conf > best_conf {
                                best_conf = conf;
                                best_class_id = Some(class_id);
                            }
                        }
                        let class_id = best_class_id?;
                        let conf = best_conf;

                        if conf < confs[idx] {
                            return None;
                        }

                        let bbox = bboxes.slice(s![idx, i, ..]);
                        let cx = (bbox[0] / ratio) * width as f32;
                        let cy = (bbox[1] / ratio) * height as f32;
                        let w = (bbox[2] / ratio) * width as f32;
                        let h = (bbox[3] / ratio) * height as f32;
                        let x = cx - w / 2.;
                        let y = cy - h / 2.;
                        let x = x.max(0.0).min(image_width as _);
                        let y = y.max(0.0).min(image_height as _);
                        let mut hbb = Hbb::default()
                            .with_xywh(x, y, w, h)
                            .with_confidence(conf)
                            .with_id(class_id as _);
                        if !names.is_empty() {
                            hbb = hbb.with_name(&names[class_id]);
                        }

                        if let Some(masks_arr) = masks {
                            let mask = masks_arr.slice(s![idx, i, .., ..]);
                            let (mh, mw) = (mask.shape()[0], mask.shape()[1]);
                            let mask = mask.into_owned().into_raw_vec_and_offset().0;
                            let mask_resized = Ops::resize_lumaf32_u8(
                                &mask,
                                mw as _,
                                mh as _,
                                image_width as _,
                                image_height as _,
                                true,
                                "Bilinear",
                            )
                            .ok()?;

                            let mask_image: image::ImageBuffer<image::Luma<_>, Vec<_>> =
                                image::ImageBuffer::from_raw(
                                    image_width as _,
                                    image_height as _,
                                    mask_resized,
                                )?;

                            let mut mask = Mask::default().with_mask(mask_image);
                            if let Some(id) = hbb.id() {
                                mask = mask.with_id(id);
                            }
                            if let Some(name) = hbb.name() {
                                mask = mask.with_name(name);
                            }
                            if let Some(confidence) = hbb.confidence() {
                                mask = mask.with_confidence(confidence);
                            }
                            Some((hbb, Some(mask)))
                        } else {
                            Some((hbb, None))
                        }
                    })
                    .collect();

                let (y_hbbs, y_masks): (Vec<_>, Vec<_>) = y_bboxes_masks.into_iter().unzip();
                let y = Y::default()
                    .with_hbbs(&y_hbbs)
                    .with_masks(&y_masks.into_iter().flatten().collect::<Vec<_>>());

                Some(y)
            })
            .collect();

        Ok(ys)
    }
}
