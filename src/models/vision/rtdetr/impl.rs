use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{
    elapsed_module, ort_inputs, Config, DynConf, Engine, FromConfig, Hbb, Image, ImageProcessor,
    Module, X, Y,
};

#[derive(Debug, Builder)]
pub struct RTDETR {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    names: Vec<String>,
    confs: DynConf,
    processor: ImageProcessor,
    spec: String,
}

impl RTDETR {
    pub fn new(mut config: Config) -> Result<Self> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&640.into()).opt(),
            engine.try_width().unwrap_or(&640.into()).opt(),
        );
        let spec = engine.spec().to_owned();
        let names: Vec<String> = config.inference.class_names.clone();
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
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<(X, X)> {
        let x1 = self.processor.process(xs)?.as_host()?;
        let x2 = X::from(vec![self.height as f32, self.width as f32])
            .insert_axis(0)?
            .repeat(0, self.batch)?;
        Ok((x1, x2))
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let (x1, x2) = elapsed_module!("RTDETR", "preprocess", self.preprocess(xs)?);

        // Extract transform info before inference
        let transform_info: Vec<_> = self
            .processor
            .images_transform_info()
            .iter()
            .map(|info| info.height_scale)
            .collect();

        // Run inference and convert to owned
        let (labels, boxes, scores) = elapsed_module!("RTDETR", "inference", {
            let ys = self.engine.run(ort_inputs![x1, x2]?)?;
            (
                X::from(ys.get::<i64>(0)?),
                X::from(ys.get::<f32>(1)?),
                X::from(ys.get::<f32>(2)?),
            )
        });

        let ys = elapsed_module!(
            "RTDETR",
            "postprocess",
            self.postprocess_impl(&labels, &boxes, &scores, &transform_info)?
        );

        Ok(ys)
    }

    fn postprocess_impl(
        &self,
        labels: &X<i64>,
        boxes: &X,
        scores: &X,
        transform_info: &[f32],
    ) -> Result<Vec<Y>> {
        let ys: Vec<Y> = labels
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(boxes.axis_iter(Axis(0)).into_par_iter())
            .zip(scores.axis_iter(Axis(0)).into_par_iter())
            .enumerate()
            .filter_map(|(idx, ((labels, boxes), scores))| {
                let ratio = transform_info[idx];
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
