use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{elapsed, DynConf, Engine, Hbb, Image, Options, Processor, Ts, Xs, X, Y};

#[derive(Debug, Builder)]
pub struct RTDETR {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    names: Vec<String>,
    confs: DynConf,
    ts: Ts,
    processor: Processor,
    spec: String,
}

impl RTDETR {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let (batch, height, width, ts) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&640.into()).opt(),
            engine.try_width().unwrap_or(&640.into()).opt(),
            engine.ts.clone(),
        );
        let spec = engine.spec().to_owned();
        let processor = options
            .to_processor()?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let names = options
            .class_names()
            .expect("No class names specified.")
            .to_vec();
        let confs = DynConf::new(options.class_confs(), names.len());

        Ok(Self {
            engine,
            height,
            width,
            batch,
            spec,
            names,
            confs,
            ts,
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
        let ys = elapsed!("preprocess", self.ts, { self.preprocess(xs)? });
        let ys = elapsed!("inference", self.ts, { self.inference(ys)? });
        let ys = elapsed!("postprocess", self.ts, { self.postprocess(ys)? });

        Ok(ys)
    }

    pub fn summary(&mut self) {
        self.ts.summary();
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

                    y_bboxes.push(
                        Hbb::default()
                            .with_xyxy(x1.max(0.0f32), y1.max(0.0f32), x2, y2)
                            .with_confidence(score)
                            .with_id(class_id)
                            .with_name(&self.names[class_id]),
                    );
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
