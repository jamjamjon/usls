use aksr::Builder;
use anyhow::Result;
use ndarray::s;

use crate::{Config, Engine, Engines, FromConfig, Image, ImageProcessor, Model, Module, Xs, X, Y};

/// Swin2SR: SwinV2 Transformer for Super-Resolution
#[derive(Debug, Builder)]
pub struct Swin2SR {
    pub batch: usize,
    pub spec: String,
    pub processor: ImageProcessor,
    pub up_scale: f32,
}

impl Model for Swin2SR {
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
        let batch = engine.batch().opt();
        let processor = ImageProcessor::from_config(config.image_processor)?;
        let up_scale = config.inference.up_scale;

        let model = Self {
            batch,
            spec,
            processor,
            up_scale,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        images
            .iter()
            .map(|image| {
                let x = crate::perf!(
                    "Swin2SR::preprocess",
                    self.processor.process(std::slice::from_ref(image))?
                );
                let ys = crate::perf!("Swin2SR::inference", engines.run(&Module::Model, &x)?);
                crate::perf!("Swin2SR::postprocess", self.postprocess_one(&ys))
            })
            .collect()
    }
}

impl Swin2SR {
    fn postprocess_one(&self, outputs: &Xs) -> Result<Y> {
        let xs = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output"))?;
        let xs = X::from(xs);
        let y = xs.permute(&[0, 2, 3, 1])?; // [b,h,w,c]
        let info = &self.processor.images_transform_info[0];
        let h = (info.height_src as f32 * self.up_scale) as usize;
        let w = (info.width_src as f32 * self.up_scale) as usize;
        let y = y.slice(s![.., 0..h, 0..w, ..]);
        let y = y.map(|x| ((x * 255.).clamp(0., 255.)) as u8);
        let image = Image::from_u8s(&y.into_raw_vec_and_offset().0, w as _, h as _)?;

        Ok(Y::default().with_images(&[image]))
    }
}
