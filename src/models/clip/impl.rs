use aksr::Builder;
use anyhow::Result;
use ndarray::Array2;

use crate::{elapsed, Config, Engine, Image, Processor, Ts, X};

#[derive(Debug, Builder)]
pub struct Clip {
    visual: Engine,
    textual: Engine,
    height: usize,
    width: usize,
    batch: usize,
    processor: Processor,
    ts: Ts,
}

impl Clip {
    pub fn new(config: Config) -> Result<Self> {
        let visual = Engine::try_from_config(&config.visual)?;
        let textual = Engine::try_from_config(&config.textual)?;
        let (batch, height, width) = (
            visual.batch().opt(),
            visual.try_height().unwrap_or(&224.into()).opt(),
            visual.try_width().unwrap_or(&224.into()).opt(),
        );
        let ts = Ts::merge(&[visual.ts(), textual.ts()]);
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            textual,
            visual,
            height,
            width,
            batch,
            processor,
            ts,
        })
    }

    pub fn encode_images(&mut self, xs: &[Image]) -> Result<X> {
        let xs = elapsed!("visual-preprocess", self.ts, {
            self.processor.process_images(xs)?
        });
        let xs = elapsed!("visual-inference", self.ts, { self.visual.run(xs.into())? });
        let x = elapsed!("visual-postprocess", self.ts, { xs[0].to_owned() });

        Ok(x)
    }

    pub fn encode_texts(&mut self, xs: &[&str]) -> Result<X> {
        let xs = elapsed!("textual-preprocess", self.ts, {
            let encodings: Vec<f32> = self
                .processor
                .encode_texts_ids(xs, false)?
                .into_iter()
                .flatten()
                .collect();

            let x: X = Array2::from_shape_vec((xs.len(), encodings.len() / xs.len()), encodings)?
                .into_dyn()
                .into();
            x
        });
        let xs = elapsed!("textual-inference", self.ts, {
            self.textual.run(xs.into())?
        });
        let x = elapsed!("textual-postprocess", self.ts, { xs[0].to_owned() });

        Ok(x)
    }

    pub fn summary(&mut self) {
        self.ts.summary();
    }
}
