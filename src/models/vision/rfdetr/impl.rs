use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{
    elapsed_module, inputs, Config, DynConf, Engine, Engines, FromConfig, Hbb, Image,
    ImageProcessor, Mask, Model, Module, Ops, Xs, Y,
};

/// RF-DETR: SOTA Real-Time Object Detection Model
#[derive(Debug, Builder)]
pub struct RFDETR {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub names: Vec<String>,
    pub confs: DynConf,
    pub classes_excluded: Vec<usize>,
    pub classes_retained: Vec<usize>,
    pub processor: ImageProcessor,
    pub spec: String,
}

impl Model for RFDETR {
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
            engine.try_height().unwrap_or(&560.into()).opt(),
            engine.try_width().unwrap_or(&560.into()).opt(),
        );
        let spec = engine.spec().to_string();
        let names: Vec<String> = config.inference.class_names;
        let confs = DynConf::new_or_default(&config.inference.class_confs, names.len());
        let classes_excluded = config.inference.classes_excluded;
        let classes_retained = config.inference.classes_retained;
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            height,
            width,
            names,
            confs,
            classes_excluded,
            classes_retained,
            processor,
            batch,
            spec,
        };

        let engines = Engines::from(engine);

        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let y = elapsed_module!("RFDETR", "preprocess", self.processor.process(images)?);
        let ys = elapsed_module!(
            "RFDETR",
            "inference",
            engines.run(&Module::Model, inputs![y]?)?
        );
        elapsed_module!("RFDETR", "postprocess", self.postprocess(&ys))
    }
}

impl RFDETR {
    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let preds_bboxes = outputs
            .get::<f32>(0)
            .ok_or(anyhow::anyhow!("Failed to get bboxes"))?;
        let preds_logits = outputs
            .get::<f32>(1)
            .ok_or(anyhow::anyhow!("Failed to get logits"))?;
        let preds_masks = outputs.get::<f32>(2);

        let ys: Vec<Y> = preds_logits
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(preds_bboxes.axis_iter(Axis(0)).into_par_iter())
            .enumerate()
            .filter_map(|(idx, (logits, bboxes))| {
                let info = &self.processor.images_transform_info[idx];
                let (image_height, image_width, ratio) =
                    (info.height_src, info.width_src, info.height_scale);
                let y_bboxes_masks: Vec<(Hbb, Option<Mask>)> = logits
                    .axis_iter(Axis(0))
                    .zip(bboxes.axis_iter(Axis(0)))
                    .enumerate()
                    .filter_map(|(i, (clss, bbox))| {
                        let (class_id, &conf) = clss.iter().enumerate().max_by(|a, b| {
                            a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                        })?;

                        let conf = 1. / ((-conf).exp() + 1.);
                        if conf < self.confs[idx] {
                            return None;
                        }

                        // filter out class id
                        if !self.classes_excluded.is_empty()
                            && self.classes_excluded.contains(&class_id)
                        {
                            return None;
                        }

                        // filter by class id
                        if !self.classes_retained.is_empty()
                            && !self.classes_retained.contains(&class_id)
                        {
                            return None;
                        }

                        // Hbb
                        let bbox = bbox.mapv(|x| x / ratio);
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

                        // Mask
                        if let Some(preds_masks) = &preds_masks {
                            let mask = preds_masks.slice(s![idx, i, .., ..]);
                            let (mh, mw) = (mask.shape()[0], mask.shape()[1]);
                            let mask = mask.into_owned().into_raw_vec_and_offset().0;

                            let mask = Ops::resize_lumaf32_u8(
                                &mask,
                                mw as _,
                                mh as _,
                                image_width as _,
                                image_height as _,
                                true,
                                "Bilinear",
                            )
                            .ok()?;

                            let mut mask =
                                Mask::new(&mask, image_width as _, image_height as _).ok()?;
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
