use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{
    elapsed_module, Config, DynConf, Engine, Hbb, Image, Mask, Ops, Processor, Task, Xs, Y,
};

#[derive(Debug, Builder)]
pub struct RFDETR {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    names: Vec<String>,
    confs: DynConf,
    processor: Processor,
    spec: String,
    task: Task,
}

impl RFDETR {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&560.into()).opt(),
            engine.try_width().unwrap_or(&560.into()).opt(),
        );
        let task = config.task().unwrap_or(&Task::ObjectDetection).clone();
        let spec = engine.spec().to_owned();
        let names: Vec<String> = config.class_names().to_vec();
        let confs = DynConf::new_or_default(config.class_confs(), names.len());
        let processor = Processor::try_from_config(&config.processor)?
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

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        let x1 = self.processor.process_images(xs)?;
        let xs = Xs::from(vec![x1]);

        Ok(xs)
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("RFDETR", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("RFDETR", "inference", self.inference(ys)?);
        let ys = elapsed_module!("RFDETR", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }

    fn postprocess(&mut self, xs: Xs) -> Result<Vec<Y>> {
        // 0: bboxes
        // 1: logits
        // 2: masks
        if xs.len() > 2 {
            self.task = Task::InstanceSegmentation;
        }

        let ys: Vec<Y> = xs[1]
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, logits)| {
                let (image_height, image_width) = (
                    self.processor.images_transform_info[idx].height_src,
                    self.processor.images_transform_info[idx].width_src,
                );
                let ratio = self.processor.images_transform_info[idx].height_scale;
                let y_bboxes_masks: Vec<(Hbb, Option<Mask>)> = logits
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .filter_map(|(i, clss)| {
                        let (class_id, &conf) = clss
                            .mapv(|x| 1. / ((-x).exp() + 1.))
                            .iter()
                            .enumerate()
                            .max_by(|a, b| {
                                a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                            })?;

                        if conf < self.confs[idx] {
                            return None;
                        }

                        let bbox = xs[0].slice(s![idx, i, ..]).mapv(|x| x / ratio);
                        let cx = bbox[0] * self.width as f32;
                        let cy = bbox[1] * self.height as f32;
                        let w = bbox[2] * self.width as f32;
                        let h = bbox[3] * self.height as f32;
                        let x = cx - w / 2.;
                        let y = cy - h / 2.;
                        let x = x.max(0.0).min(image_width as _);
                        let y = y.max(0.0).min(image_height as _);
                        let mut hbb = Hbb::default()
                            .with_xywh(x, y, w, h)
                            .with_confidence(conf)
                            .with_id(class_id as _);
                        if !self.names.is_empty() {
                            hbb = hbb.with_name(&self.names[class_id]);
                        }

                        if matches!(self.task, Task::InstanceSegmentation) {
                            let mask = xs[2].slice(s![idx, i, .., ..]);
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
