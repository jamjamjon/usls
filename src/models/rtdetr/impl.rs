use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{elapsed_module, Config, DynConf, Engine, Hbb, Image, Processor, Xs, X, Y};

#[derive(Debug, Builder)]
pub struct RTDETR {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    names: Vec<String>,
    confs: DynConf,
    processor: Processor,
    spec: String,
}

impl RTDETR {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&640.into()).opt(),
            engine.try_width().unwrap_or(&640.into()).opt(),
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
        let x2 = X::from(vec![self.height as f32, self.width as f32])
            .insert_axis(0)?
            .repeat(0, self.batch)?;

        let xs = Xs::from(vec![x1, x2]);

        Ok(xs)
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("RTDETR", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("RTDETR", "inference", self.inference(ys)?);
        let ys = elapsed_module!("RTDETR", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }

    fn postprocess(&mut self, xs: Xs) -> Result<Vec<Y>> {
        let ys: Vec<Y> = xs[0]
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(xs[1].axis_iter(Axis(0)).into_par_iter())
            .zip(xs[2].axis_iter(Axis(0)).into_par_iter())
            .enumerate()
            .filter_map(|(idx, ((labels, boxes), scores))| {
                let ratio = self.processor.images_transform_info[idx].height_scale;
                let mut y_bboxes = Vec::new();
                for (i, &score) in scores.iter().enumerate() {
                    let class_id = labels[i] as usize;
                    if score < self.confs[class_id] {
                        continue;
                    }
                    let xyxy = boxes.slice(s![i, ..]);
                    let (x1, y1, x2, y2) = (
                        xyxy[0] / ratio,
                        xyxy[1] / ratio,
                        xyxy[2] / ratio,
                        xyxy[3] / ratio,
                    );
                    let mut hbb = Hbb::default()
                        .with_xyxy(x1.max(0.0f32), y1.max(0.0f32), x2, y2)
                        .with_confidence(score)
                        .with_id(class_id);
                    if !self.names.is_empty() {
                        hbb = hbb.with_name(&self.names[class_id]);
                    }
                    y_bboxes.push(hbb);
                }

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
