use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{
    elapsed_module, inputs, Config, DynConf, Engine, Engines, FromConfig, Image, ImageProcessor,
    Model, Module, Text, Xs, X, Y,
};

/// RAM: Recognize Anything Model
#[derive(Debug, Builder)]
pub struct Ram {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub confs: DynConf,
    pub spec: String,
    pub processor: ImageProcessor,
    pub names_zh: Vec<String>,
    pub names_en: Vec<String>,
}

impl Model for Ram {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let (batch, height, width, spec) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&384.into()).opt(),
            engine.try_width().unwrap_or(&384.into()).opt(),
            engine.spec().to_owned(),
        );
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let names_zh = config.inference.class_names;
        let names_en = config.inference.class_names2;
        let nc = names_zh.len();
        let confs = DynConf::new_or_default(&config.inference.class_confs, nc);

        let model = Self {
            confs,
            height,
            width,
            batch,
            processor,
            spec,
            names_zh,
            names_en,
        };

        Ok((model, engine.into()))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let x = elapsed_module!("RAM", "preprocess", self.processor.process(images)?);
        let ys = elapsed_module!(
            "RAM",
            "inference",
            engines.run(&Module::Model, inputs![x]?)?
        );
        elapsed_module!("RAM", "postprocess", self.postprocess(&ys))
    }
}

impl Ram {
    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let xs = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output"))?;
        let xs = X::from(xs);
        let ys: Vec<Y> = xs
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(_batch_idx, logits)| {
                let texts: Vec<Text> = logits
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &x)| {
                        let conf = 1.0 / (1.0 + (-x).exp());
                        if conf < self.confs[i] {
                            return None;
                        }
                        Some(Text::from(format!(
                            "{}({})",
                            self.names_zh[i], self.names_en[i]
                        )))
                    })
                    .collect();

                if texts.is_empty() {
                    return None;
                }

                Some(Y::default().with_texts(&texts))
            })
            .collect();

        Ok(ys)
    }
}
