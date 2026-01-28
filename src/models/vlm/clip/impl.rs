use aksr::Builder;
use anyhow::Result;

use crate::{
    elapsed_module, inputs, Config, Engine, Engines, FromConfig, Image, ImageProcessor, Model,
    Module, TextProcessor, X, Y,
};

/// CLIP: Contrastive Language-Image Pre-Training
#[derive(Debug, Builder)]
pub struct Clip {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub image_processor: ImageProcessor,
    pub text_processor: TextProcessor,
    pub spec: String,
}

impl Model for Clip {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let visual = Engine::from_config(config.take_module(&Module::Visual)?)?;
        let textual = Engine::from_config(config.take_module(&Module::Textual)?)?;
        let (batch, height, width) = (
            visual.batch().opt(),
            visual.try_height().unwrap_or(&224.into()).opt(),
            visual.try_width().unwrap_or(&224.into()).opt(),
        );
        let spec = visual.spec().to_owned();

        let image_processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        #[cfg(feature = "vlm")]
        let text_processor = TextProcessor::from_config(config.text_processor)?;
        #[cfg(not(feature = "vlm"))]
        let text_processor = TextProcessor::default();

        let model = Self {
            height,
            width,
            batch,
            image_processor,
            text_processor,
            spec,
        };

        let mut engines = Engines::default();
        engines.insert(Module::Visual, visual);
        engines.insert(Module::Textual, textual);
        Ok((model, engines))
    }

    fn encode_images(&mut self, engines: &mut Engines, images: &[Image]) -> Result<Y> {
        let ys = elapsed_module!("CLIP", "visual-preprocess", {
            self.image_processor.process(images)?
        });
        let ys = elapsed_module!(
            "CLIP",
            "visual-inference",
            engines.run(&Module::Visual, &ys)?
        );
        let y = elapsed_module!(
            "CLIP",
            "visual-postprocess",
            ys.get::<f32>(0)
                .ok_or_else(|| anyhow::anyhow!("Failed to get visual output"))?
                .to_owned()
        );

        Ok(Y::default().with_embedding(y))
    }

    fn encode_texts(&mut self, engines: &mut Engines, texts: &[&str]) -> Result<Y> {
        let ys = elapsed_module!("CLIP", "textual-preprocess", {
            let encodings: Vec<f32> = self
                .text_processor
                .encode_texts_ids(texts, true)?
                .into_iter()
                .flatten()
                .collect();
            let shape = &[texts.len(), encodings.len() / texts.len()];
            X::from_shape_vec(shape, encodings)?
        });
        let ys = elapsed_module!(
            "CLIP",
            "textual-inference",
            engines.run(&Module::Textual, inputs![ys]?)?
        );
        let y = elapsed_module!(
            "CLIP",
            "textual-postprocess",
            ys.get::<f32>(0)
                .ok_or_else(|| anyhow::anyhow!("Failed to get textual output"))?
                .to_owned()
        );

        Ok(Y::default().with_embedding(y))
    }
}
