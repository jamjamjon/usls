use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{
    elapsed_module, ort_inputs, Config, DynConf, Engine, FromConfig, Image, ImageProcessor, Module,
    Text, X, Y,
};

#[derive(Debug, Builder)]
pub struct Ram {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    confs: DynConf,
    spec: String,
    processor: ImageProcessor,
    names_zh: Vec<String>,
    names_en: Vec<String>,
}

impl Ram {
    pub fn new(mut config: Config) -> Result<Self> {
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
        let names_zh = config.inference.class_names.clone();
        let names_en = config.inference.class_names2.clone();
        let nc = names_zh.len();
        let confs = DynConf::new_or_default(&config.inference.class_confs, nc);

        Ok(Self {
            engine,
            confs,
            height,
            width,
            batch,
            processor,
            spec,
            names_zh,
            names_en,
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<X> {
        self.processor.process(xs)?.as_host()
    }

    fn inference(&mut self, xs: X) -> Result<X> {
        let output = self.engine.run(ort_inputs![xs]?)?;
        Ok(X::from(output.get::<f32>(0)?))
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("RAM", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("RAM", "inference", self.inference(ys)?);
        let ys = elapsed_module!("RAM", "postprocess", self.postprocess(&ys)?);

        Ok(ys)
    }

    pub fn postprocess(&mut self, xs: &X) -> Result<Vec<Y>> {
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
