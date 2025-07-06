use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{elapsed_module, Config, DynConf, Engine, Image, Processor, Text, Xs, Y};

/// SVTR (Scene Text Recognition) model for text recognition.
#[derive(Builder, Debug)]
pub struct SVTR {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    confs: DynConf,
    spec: String,
    processor: Processor,
}

impl SVTR {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&960.into()).opt(),
            engine.try_width().unwrap_or(&960.into()).opt(),
        );
        let spec = config.model.spec.to_string();
        let confs = DynConf::new_or_default(config.class_confs(), 1);
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        if processor.vocab().is_empty() {
            anyhow::bail!("No vocab file found")
        }
        log::info!("Vacab size: {}", processor.vocab().len());

        Ok(Self {
            engine,
            height,
            width,
            batch,
            confs,
            processor,
            spec,
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        Ok(self.processor.process_images(xs)?.into())
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("SVTR", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("SVTR", "inference", self.inference(ys)?);
        let ys = elapsed_module!("SVTR", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }

    pub fn postprocess(&self, xs: Xs) -> Result<Vec<Y>> {
        let ys: Vec<Y> = xs[0]
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|preds| {
                let mut preds: Vec<_> = preds
                    .axis_iter(Axis(0))
                    .filter_map(|x| x.into_iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)))
                    .collect();

                preds.dedup_by(|a, b| a.0 == b.0);

                let (text, confs): (String, Vec<f32>) = preds
                    .into_iter()
                    .filter(|(id, &conf)| *id != 0 && conf >= self.confs[0])
                    .map(|(id, &conf)| (self.processor.vocab()[id].clone(), conf))
                    .collect();

                Y::default().with_texts(&[Text::from(text)
                    .with_confidence(confs.iter().sum::<f32>() / confs.len() as f32)])
            })
            .collect();

        Ok(ys)
    }
}
