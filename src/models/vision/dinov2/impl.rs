use aksr::Builder;
use anyhow::Result;

use crate::{
    elapsed_module, ort_inputs, Config, Engine, FromConfig, Image, ImageProcessor, Module, X,
};

#[derive(Builder, Debug)]
pub struct DINOv2 {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    processor: ImageProcessor,
}

impl DINOv2 {
    pub fn new(mut config: Config) -> Result<Self> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&384.into()).opt(),
            engine.try_width().unwrap_or(&384.into()).opt(),
        );
        let processor = ImageProcessor::from_config(config.image_processor)?
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

    fn preprocess(&mut self, xs: &[Image]) -> Result<X> {
        self.processor.process(xs)?.as_host()
    }

    pub fn encode_images(&mut self, xs: &[Image]) -> Result<X> {
        let x = elapsed_module!("DINOv2", "visual-preprocess", self.preprocess(xs)?);

        let output = elapsed_module!("DINOv2", "visual-inference", {
            let ys = self.engine.run(ort_inputs![x]?)?;
            X::from(ys.get::<f32>(0)?)
        });

        Ok(output)
    }
}
