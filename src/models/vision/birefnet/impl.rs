use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{
    elapsed_module, inputs, Config, Engine, Engines, FromConfig, Image, ImageProcessor, Mask,
    Model, Module, Ops, Xs, Y,
};

/// MODNet: Trimap-Free Portrait Matting in Real Time
#[derive(Builder, Debug)]
pub struct BiRefNet {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub spec: String,
    pub processor: ImageProcessor,
}

impl Model for BiRefNet {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let spec = engine.spec().to_string();
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&512.into()).opt(),
            engine.try_width().unwrap_or(&512.into()).opt(),
        );
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            height,
            width,
            batch,
            spec,
            processor,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let x = elapsed_module!("BiRefNet", "preprocess", self.processor.process(images)?);
        let ys = elapsed_module!(
            "BiRefNet",
            "inference",
            engines.run(&Module::Model, inputs![&x]?)?
        );
        elapsed_module!("BiRefNet", "postprocess", self.postprocess(&ys))
    }
}

impl BiRefNet {
    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let xs = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output"))?;
        let ys: Vec<Y> = xs
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, luma)| {
                let info = &self.processor.images_transform_info[idx];
                let (h1, w1) = (info.height_src, info.width_src);
                let luma = luma.mapv(|x| (1. / ((-x).exp() + 1.) * 255.0) as u8);
                let luma = Ops::resize_luma8_u8(
                    &luma.into_raw_vec_and_offset().0,
                    self.width as _,
                    self.height as _,
                    w1 as _,
                    h1 as _,
                    false,
                    "Bilinear",
                )
                .ok()?;

                image::ImageBuffer::from_raw(w1 as _, h1 as _, luma)
                    .map(|luma| Y::default().with_masks(&[Mask::default().with_mask(luma)]))
            })
            .collect();

        Ok(ys)
    }
}
