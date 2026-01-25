use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{
    elapsed_module, inputs, Config, Engine, Engines, FromConfig, Image, ImageProcessor, Mask,
    Model, Module, Ops, Xs, Y,
};

/// Depth Pro: Sharp Monocular Metric Depth
#[derive(Builder, Debug)]
pub struct DepthPro {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub spec: String,
    pub processor: ImageProcessor,
}

impl Model for DepthPro {
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
        let x = elapsed_module!("DepthPro", "preprocess", self.processor.process(images)?);
        let ys = elapsed_module!(
            "DepthPro",
            "inference",
            engines.run(&Module::Model, inputs![&x]?)?
        );
        elapsed_module!("DepthPro", "postprocess", self.postprocess(&ys))
    }
}

impl DepthPro {
    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let depth = outputs
            .get::<f32>(0)
            // .get_by_name::<f32>("predicted_depth")
            .ok_or_else(|| anyhow::anyhow!("Failed to get predicted_depth"))?;
        let predicted_depth = depth.mapv(|x| 1. / x);

        let ys: Vec<Y> = predicted_depth
            .axis_iter(Axis(0))
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
