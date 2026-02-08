use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{
    Config, Engine, Engines, FromConfig, Image, ImageProcessor, Model, Module, Prob, Xs, Y,
};

/// ImageClassifier - A classification model implementing the Model trait.
///
/// This model provides image classification with support for softmax,
/// top-k predictions, and class names. It uses a single engine for
/// inference and returns probability distributions over classes.
#[derive(Debug, Builder)]
pub struct ImageClassifier {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub apply_softmax: bool,
    pub processor: ImageProcessor,
    pub names: Vec<String>,
    pub spec: String,
    pub topk: usize,
}

impl ImageClassifier {
    fn postprocess(&self, xs: &Xs) -> Result<Vec<Y>> {
        let xs = xs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output"))?;

        let ys: Vec<Y> = xs
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

impl Model for ImageClassifier {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let spec = engine.spec().to_string();
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&224.into()).opt(),
            engine.try_width().unwrap_or(&224.into()).opt(),
        );
        let names = config.inference.class_names;
        let apply_softmax = config.inference.apply_softmax.unwrap_or_default();
        let topk = config.inference.topk.unwrap_or(5);
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            height,
            width,
            batch,
            spec,
            processor,
            names,
            apply_softmax,
            topk,
        };

        Ok((model, engine.into()))
    }

    fn run(&mut self, engines: &mut Engines, input: Self::Input<'_>) -> Result<Vec<Y>> {
        let xs = crate::perf!(
            "Image-Classifier::preprocess",
            self.processor.process(input)?
        );
        let ys = crate::perf!(
            "Image-Classifier::inference",
            engines.run(&Module::Model, &xs)?
        );
        let ys = crate::perf!("Image-Classifier::postprocess", self.postprocess(&ys)?);

        Ok(ys)
    }
}
