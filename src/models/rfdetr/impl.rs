use aksr::Builder;
use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{elapsed, Bbox, DynConf, Engine, Options, Processor, Ts, Xs, Ys, Y};

#[derive(Debug, Builder)]
pub struct RFDETR {
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

impl RFDETR {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let (batch, height, width, ts) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&560.into()).opt(),
            engine.try_width().unwrap_or(&560.into()).opt(),
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

    fn preprocess(&mut self, xs: &[DynamicImage]) -> Result<Xs> {
        let x1 = self.processor.process_images(xs)?;
        let xs = Xs::from(vec![x1]);

        Ok(xs)
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[DynamicImage]) -> Result<Ys> {
        let ys = elapsed!("preprocess", self.ts, { self.preprocess(xs)? });
        let ys = elapsed!("inference", self.ts, { self.inference(ys)? });
        let ys = elapsed!("postprocess", self.ts, { self.postprocess(ys)? });

        Ok(ys)
    }

    fn postprocess(&mut self, xs: Xs) -> Result<Ys> {
        // 0: bboxes
        // 1: logits
        let ys: Vec<Y> = xs[1]
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, logits)| {
                let (image_height, image_width) = self.processor.image0s_size[idx];
                let ratio = self.processor.scale_factors_hw[idx][0];
                let y_bboxes: Vec<Bbox> = logits
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

                        Some(
                            Bbox::default()
                                .with_xywh(x, y, w, h)
                                .with_confidence(conf)
                                .with_id(class_id as _)
                                .with_name(&self.names[class_id]),
                        )
                    })
                    .collect();

                let mut y = Y::default();
                if !y_bboxes.is_empty() {
                    y = y.with_bboxes(&y_bboxes);
                }

                Some(y)
            })
            .collect();

        Ok(ys.into())
    }

    pub fn summary(&mut self) {
        self.ts.summary();
    }
}
