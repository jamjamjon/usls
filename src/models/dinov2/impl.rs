use aksr::Builder;
use anyhow::Result;

use crate::{elapsed_module, Config, Engine, Image, Processor, Xs, X};

#[derive(Builder, Debug)]
pub struct DINOv2 {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
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
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            engine,
            height,
            width,
            batch,
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
