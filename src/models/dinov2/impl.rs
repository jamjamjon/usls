use aksr::Builder;
use anyhow::Result;

use crate::{elapsed_module, Config, Engine, Image, Processor, Scale, Xs, X};

#[derive(Builder, Debug)]
pub struct DINOv2 {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    dim: usize,
    processor: Processor,
}

impl DINOv2 {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&384.into()).opt(),
            engine.try_width().unwrap_or(&384.into()).opt(),
        );
        let dim = match &config.scale {
            Some(Scale::S) => 384,
            Some(Scale::B) => 768,
            Some(Scale::L) => 1024,
            Some(Scale::G) => 1536,
            Some(x) => anyhow::bail!("Unsupported scale: {:?}", x),
            None => anyhow::bail!("No model scale specified"),
        };
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            engine,
            height,
            width,
            batch,
            dim,
            processor,
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        let x = self.processor.process_images(xs)?;

        Ok(x.into())
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn encode_images(&mut self, xs: &[Image]) -> Result<X> {
        let xs = elapsed_module!("DINOv2", "visual-preprocess", self.preprocess(xs)?);
        let xs = elapsed_module!("DINOv2", "visual-inference", self.inference(xs)?);
        let x = elapsed_module!("DINOv2", "visual-postprocess", xs[0].to_owned());

        Ok(x)
    }
}
