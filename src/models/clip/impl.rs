use aksr::Builder;
use anyhow::Result;
use ndarray::Array2;

use crate::{elapsed_module, Config, Engine, Image, Processor, X};

#[derive(Debug, Builder)]
pub struct Clip {
    visual: Engine,
    textual: Engine,
    height: usize,
    width: usize,
    batch: usize,
    processor: Processor,
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
        })
    }

    pub fn encode_images(&mut self, xs: &[Image]) -> Result<X> {
        let xs = elapsed_module!("CLIP", "visual-preprocess", {
            self.processor.process_images(xs)?
        });
        let xs = elapsed_module!("CLIP", "visual-inference", self.visual.run(xs.into())?);
        let x = elapsed_module!("CLIP", "visual-postprocess", xs[0].to_owned());

        Ok(x)
    }

    pub fn encode_texts(&mut self, xs: &[&str]) -> Result<X> {
        let xs = elapsed_module!("CLIP", "textual-preprocess", {
            let encodings: Vec<f32> = self
                .processor
                .encode_texts_ids(xs, true)?
                .into_iter()
                .flatten()
                .collect();

            let x: X = Array2::from_shape_vec((xs.len(), encodings.len() / xs.len()), encodings)?
                .into_dyn()
                .into();

            x
        });
        let xs = elapsed_module!("CLIP", "textual-inference", {
            self.textual.run(xs.into())?
        });
        let x = elapsed_module!("CLIP", "textual-postprocess", xs[0].to_owned());

        Ok(x)
    }
}
