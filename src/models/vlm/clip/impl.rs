use aksr::Builder;
use anyhow::Result;

use crate::{elapsed_module, Config, Engine, FromConfig, Image, ImageProcessor, TextProcessor, X};

#[derive(Debug, Builder)]
pub struct Clip {
    visual: Engine,
    textual: Engine,
    height: usize,
    width: usize,
    batch: usize,
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
}

impl Clip {
    pub fn new(mut config: Config) -> Result<Self> {
        let visual = Engine::from_config(config.take_module(&crate::Module::Visual)?)?;
        let textual = Engine::from_config(config.take_module(&crate::Module::Textual)?)?;
        let (batch, height, width) = (
            visual.batch().opt(),
            visual.try_height().unwrap_or(&224.into()).opt(),
            visual.try_width().unwrap_or(&224.into()).opt(),
        );

        let image_processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        #[cfg(feature = "vlm")]
        let text_processor = TextProcessor::from_config(config.text_processor)?;
        #[cfg(not(feature = "vlm"))]
        let text_processor = TextProcessor::default();

        Ok(Self {
            textual,
            visual,
            height,
            width,
            batch,
            image_processor,
            text_processor,
        })
    }

    pub fn encode_images(&mut self, xs: &[Image]) -> Result<X> {
        let ys = elapsed_module!("CLIP", "visual-preprocess", {
            self.image_processor.process(xs)?
        });
        let ys = elapsed_module!(
            "CLIP",
            "visual-inference",
            self.visual.run(ort_inputs![ys]?)?
        );
        let y = elapsed_module!("CLIP", "visual-postprocess", ys.get::<f32>(0)?.to_owned());

        Ok(y)
    }

    pub fn encode_texts(&mut self, xs: &[&str]) -> Result<X> {
        let ys = elapsed_module!("CLIP", "textual-preprocess", {
            let encodings: Vec<f32> = self
                .text_processor
                .encode_texts_ids(xs, true)?
                .into_iter()
                .flatten()
                .collect();
            let shape = &[xs.len(), encodings.len() / xs.len()];
            X::from_shape_vec(shape, encodings)?
        });
        let ys = elapsed_module!(
            "CLIP",
            "textual-inference",
            self.textual.run(ort_inputs![ys]?)?
        );
        let y = elapsed_module!("CLIP", "textual-postprocess", ys.get::<f32>(0)?.to_owned());

        Ok(y)
    }
}
