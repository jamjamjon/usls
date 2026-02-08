use aksr::Builder;
use anyhow::Result;
use rayon::prelude::*;

use crate::{
    Config, Engine, Engines, FromConfig, Image, ImageProcessor, Mask, Model, Module, Ops, Xs, Y,
};

/// RMBG: BRIA Background Removal
#[derive(Builder, Debug)]
pub struct RMBG {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub spec: String,
    pub processor: ImageProcessor,
}

impl Model for RMBG {
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
            engine.try_height().unwrap_or(&1024.into()).opt(),
            engine.try_width().unwrap_or(&1024.into()).opt(),
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
        let x = crate::perf!("RMBG::preprocess", self.processor.process(images)?);
        let ys = crate::perf!("RMBG::inference", engines.run(&Module::Model, &x)?);
        crate::perf!("RMBG::postprocess", self.postprocess(&ys))
    }
}

impl RMBG {
    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let output = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output"))?;

        let ys: Vec<Y> = output
            .axis_iter(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, luma)| {
                let info = &self.processor.images_transform_info[idx];
                let (h1, w1) = (info.height_src, info.width_src);
                let v = luma.into_owned().into_raw_vec_and_offset().0;
                let max_ = v.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
                let min_ = v.iter().min_by(|x, y| x.total_cmp(y)).unwrap();
                let v: Vec<f32> = v
                    .par_iter()
                    .map(|x| ((*x - min_) / (max_ - min_)).clamp(0., 1.))
                    .collect();

                Ops::interpolate_1d_u8(
                    &v,
                    self.width as _,
                    self.height as _,
                    w1 as _,
                    h1 as _,
                    false,
                )
                .ok()
                .and_then(|luma| Mask::new(&luma, w1, h1).ok())
                .map(|mask| Y::default().with_masks(&[mask]))
            })
            .collect();

        Ok(ys)
    }
}
