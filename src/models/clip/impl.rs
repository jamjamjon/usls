use aksr::Builder;
use anyhow::Result;
use ndarray::Array2;

use crate::{elapsed, CLIPConfig, Engine, Image, Processor, Ts, X};

#[derive(Debug, Builder)]
pub struct CLIP {
    textual: Engine,
    visual: Engine,
    height: usize,
    width: usize,
    batch: usize,
    processor: Processor,
    ts: Ts,
}

impl CLIP {
    pub fn new(config: CLIPConfig) -> Result<Self> {
        let visual = Engine::try_from(config.visual)?;
        let textual = Engine::try_from(config.textual)?;
        let (batch, height, width, ts, _spec) = (
            visual.batch().opt().min(textual.batch().opt()),
            visual.try_height().unwrap_or(&224.into()).opt(),
            visual.try_width().unwrap_or(&224.into()).opt(),
            Ts::default(),
            visual.spec().to_owned(),
        );

        let processor = Processor::try_from_config(config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            textual,
            visual,
            height,
            width,
            batch,
            ts,
            processor,
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
                .encode_texts_ids(xs, false)? // skip_special_tokens
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
        self.ts = Ts::merge(&[self.visual.ts(), self.textual.ts(), &self.ts]);
        self.ts.summary();
    }
}
