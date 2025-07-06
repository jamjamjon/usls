use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{elapsed_module, Config, DynConf, Engine, Hbb, Image, Processor, Xs, Y};

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
}

impl RFDETR {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&560.into()).opt(),
            engine.try_width().unwrap_or(&560.into()).opt(),
        );
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
                let y_bboxes: Vec<Hbb> = logits
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .filter_map(|(i, clss)| {
                        let (class_id, &conf) = clss
                            .mapv(|x| 1. / ((-x).exp() + 1.))
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))?;

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

                        Some(hbb)
                    })
                    .collect();

                let mut y = Y::default();
                if !y_bboxes.is_empty() {
                    y = y.with_hbbs(&y_bboxes);
                }

                Some(y)
            })
            .collect();

        Ok(ys)
    }
}
