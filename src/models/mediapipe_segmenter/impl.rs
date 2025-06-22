use aksr::Builder;
use anyhow::Result;
use rayon::prelude::*;

use crate::{elapsed_module, Config, Engine, Image, Mask, Ops, Processor, Xs, Y};

#[derive(Builder, Debug)]
pub struct MediaPipeSegmenter {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    spec: String,
    processor: Processor,
}

impl MediaPipeSegmenter {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let spec = engine.spec().to_string();
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&256.into()).opt(),
            engine.try_width().unwrap_or(&256.into()).opt(),
        );
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            engine,
            height,
            width,
            batch,
            spec,
            processor,
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        Ok(self.processor.process_images(xs)?.into())
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("mediapipe_segmenter", "preprocess", {
            self.preprocess(xs)?
        });
        let ys = elapsed_module!("mediapipe_segmenter", "inference", self.inference(ys)?);
        let ys = elapsed_module!("mediapipe_segmenter", "postprocess", {
            self.postprocess(ys)?
        });

        Ok(ys)
    }
    fn postprocess(&mut self, xs: Xs) -> Result<Vec<Y>> {
        let ys: Vec<Y> = xs[0]
            .iter_dim(0)
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, luma)| {
                let (h1, w1) = (
                    self.processor.images_transform_info[idx].height_src,
                    self.processor.images_transform_info[idx].width_src,
                );

                // Convert tensor to Vec<f32> and apply scaling
                let luma_vec: Vec<f32> = luma.to_vec().ok()?;
                let luma_u8: Vec<u8> = luma_vec.iter().map(|&x| (x * 255.0) as u8).collect();

                let resized_luma = Ops::resize_luma8_u8(
                    &luma_u8,
                    self.width as _,
                    self.height as _,
                    w1 as _,
                    h1 as _,
                    false,
                    "Bilinear",
                )
                .ok()?;

                let luma_buffer: image::ImageBuffer<image::Luma<_>, Vec<_>> =
                    image::ImageBuffer::from_raw(w1 as _, h1 as _, resized_luma)?;

                Some(Y::default().with_masks(&[Mask::default().with_mask(luma_buffer)]))
            })
            .collect();

        Ok(ys)
    }
}
