use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{elapsed_module, Config, Engine, Image, Prob, Processor, Xs, Y};

#[derive(Debug, Builder)]
pub struct ImageClassifier {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    apply_softmax: bool,
    processor: Processor,
    names: Vec<String>,
    spec: String,
    topk: usize,
}

impl TryFrom<Config> for ImageClassifier {
    type Error = anyhow::Error;

    fn try_from(config: Config) -> Result<Self, Self::Error> {
        Self::new(config)
    }
}

impl ImageClassifier {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let spec = engine.spec().to_string();
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&224.into()).opt(),
            engine.try_width().unwrap_or(&224.into()).opt(),
        );
        let names = config.class_names.to_vec();
        let apply_softmax = config.apply_softmax.unwrap_or_default();
        let topk = config.topk.unwrap_or(5);
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            engine,
            height,
            width,
            batch,
            spec,
            processor,
            names,
            apply_softmax,
            topk,
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        let x = self.processor.process_images(xs)?;

        Ok(x.into())
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("ImageClassifier", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("ImageClassifier", "inference", self.inference(ys)?);
        let ys = elapsed_module!("ImageClassifier", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }

    fn postprocess(&self, xs: Xs) -> Result<Vec<Y>> {
        let ys: Vec<Y> = xs[0]
            .axis_iter(Axis(0))
            .into_par_iter()
            .filter_map(|logits| {
                let logits = if self.apply_softmax {
                    let exps = logits.mapv(|x| x.exp());
                    let stds = exps.sum_axis(Axis(0));
                    exps / stds
                } else {
                    logits.into_owned()
                };
                let probs = Prob::new_probs(
                    &logits.into_raw_vec_and_offset().0,
                    Some(&self.names.iter().map(|x| x.as_str()).collect::<Vec<_>>()),
                    self.topk,
                );

                Some(Y::default().with_probs(&probs))
            })
            .collect::<Vec<_>>();

        Ok(ys)
    }
}
