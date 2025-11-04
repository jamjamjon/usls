use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{elapsed_module, Config, DynConf, Engine, Image, Processor, Text, Xs, Y};

#[derive(Debug, Builder)]
pub struct Ram {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    confs: DynConf,
    spec: String,
    processor: Processor,
    names_zh: Vec<String>,
    names_en: Vec<String>,
}

impl Ram {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let (batch, height, width, spec) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&384.into()).opt(),
            engine.try_width().unwrap_or(&384.into()).opt(),
            engine.spec().to_owned(),
        );
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let names_zh = config.class_names().to_vec();
        let names_en = config.class_names2().to_vec();
        let nc = names_zh.len();
        let confs = DynConf::new_or_default(config.class_confs(), nc);

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

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        Ok(self.processor.process_images(xs)?.into())
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("RAM", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("RAM", "inference", self.inference(ys)?);
        let ys = elapsed_module!("RAM", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }

    pub fn postprocess(&mut self, xs: Xs) -> Result<Vec<Y>> {
        let ys: Vec<Y> = xs[0]
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
