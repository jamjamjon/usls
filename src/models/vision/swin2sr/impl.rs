use aksr::Builder;
use anyhow::Result;
use ndarray::s;

use crate::{
    elapsed_module, ort_inputs, Config, Engine, FromConfig, Image, ImageProcessor, Module, X, Y,
};

#[derive(Debug, Builder)]
pub struct Swin2SR {
    engine: Engine,
    batch: usize,
    spec: String,
    processor: ImageProcessor,
    up_scale: f32,
}

impl Swin2SR {
    pub fn new(mut config: Config) -> Result<Self> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let spec = engine.spec().to_string();
        let batch = engine.batch().opt();
        let up_scale = config.image_processor.up_scale;
        let processor = ImageProcessor::from_config(config.image_processor)?;

        Ok(Self {
            engine,
            batch,
            spec,
            processor,
            up_scale,
        })
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        xs.iter()
            .map(|x| {
                let y = elapsed_module!("Swin2SR", "preprocess_one", self.preprocess_one(x)?);
                let y = elapsed_module!("Swin2SR", "inference", self.inference(y)?);
                elapsed_module!("Swin2SR", "postprocess_one", self.postprocess_one(y))
            })
            .collect()
    }

    fn preprocess_one(&mut self, xs: &Image) -> Result<X> {
        self.processor.process(&[xs.clone()])?.as_host()
    }

    fn inference(&mut self, xs: X) -> Result<X> {
        let output = self.engine.run(ort_inputs![xs]?)?;
        Ok(X::from(output.get::<f32>(0)?))
    }

    fn postprocess_one(&mut self, xs: X) -> Result<Y> {
        let y = xs.permute(&[0, 2, 3, 1])?; // [b,h,w,c]
        let h =
            (self.processor.images_transform_info()[0].height_src as f32 * self.up_scale) as usize;
        let w =
            (self.processor.images_transform_info()[0].width_src as f32 * self.up_scale) as usize;
        let y = y.slice(s![.., 0..h, 0..w, ..]);
        let y = y.map(|x| ((x * 255.).clamp(0., 255.)) as u8);
        let image = Image::from_u8s(&y.into_raw_vec_and_offset().0, w as _, h as _)?;

        Ok(Y::default().with_images(&[image]))
    }
}
