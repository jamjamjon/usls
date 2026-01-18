use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{
    elapsed_module, inputs, Config, DynConf, Engine, Engines, FromConfig, Image, ImageProcessor,
    Model, Module, Text, Xs, X, Y,
};

/// SVTR (Scene Text Recognition) model for text recognition.
#[derive(Builder, Debug)]
pub struct SVTR {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub confs: DynConf,
    pub spec: String,
    pub processor: ImageProcessor,
    pub vocab: Vec<String>,
}

impl Model for SVTR {
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
            engine.try_height().unwrap_or(&960.into()).opt(),
            engine.try_width().unwrap_or(&960.into()).opt(),
        );
        let spec = engine.spec().to_string();
        let confs = DynConf::new_or_default(&config.inference.class_confs, 1);
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let vocab = config.inference.class_names;
        tracing::info!("Vacab size: {}", vocab.len());

        let model = Self {
            height,
            width,
            batch,
            confs,
            processor,
            spec,
            vocab,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let x = elapsed_module!("SVTR", "preprocess", self.processor.process(images)?);
        let ys = elapsed_module!(
            "SVTR",
            "inference",
            engines.run(&Module::Model, inputs![&x]?)?
        );
        elapsed_module!("SVTR", "postprocess", self.postprocess(&ys))
    }
}

impl SVTR {
    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let xs = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output"))?;
        let xs = X::from(xs);
        let ys: Vec<Y> = xs
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
                    .map(|(id, &conf)| (self.vocab[id].clone(), conf))
                    .collect();

                Y::default().with_texts(&[Text::from(text)
                    .with_confidence(confs.iter().sum::<f32>() / confs.len() as f32)])
            })
            .collect();

        Ok(ys)
    }
}
