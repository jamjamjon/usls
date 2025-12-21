use aksr::Builder;
use anyhow::Result;

use crate::{
    elapsed_module, ort_inputs, Config, Engine, FromConfig, Image, ImageProcessor,
    ImageTransformInfo, Mask, Module, Ops, XView, X, Y,
};

#[derive(Builder, Debug)]
pub struct RMBG {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    spec: String,
    processor: ImageProcessor,
}

impl RMBG {
    pub fn new(mut config: Config) -> Result<Self> {
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

        Ok(Self {
            engine,
            height,
            width,
            batch,
            spec,
            processor,
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<X> {
        self.processor.process(xs)?.as_host()
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let x = elapsed_module!("RMBG", "preprocess", self.preprocess(xs)?);

        let results = {
            let ys = elapsed_module!("RMBG", "inference", self.engine.run(ort_inputs![x]?)?);

            let output = ys.get::<f32>(0)?;

            elapsed_module!(
                "RMBG",
                "postprocess",
                Self::postprocess_impl(
                    self.processor.images_transform_info(),
                    self.width,
                    self.height,
                    &output,
                )?
            )
        };

        Ok(results)
    }

    fn postprocess_impl(
        images_transform_info: &[ImageTransformInfo],
        width: usize,
        height: usize,
        output: &XView<'_, f32>,
    ) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        for (idx, luma) in output.axis_iter(ndarray::Axis(0)).enumerate() {
            let (h1, w1) = (
                images_transform_info[idx].height_src,
                images_transform_info[idx].width_src,
            );
            let v = luma.into_owned().into_raw_vec_and_offset().0;
            let max_ = v.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
            let min_ = v.iter().min_by(|x, y| x.total_cmp(y)).unwrap();
            let v = v
                .iter()
                .map(|x| (((*x - min_) / (max_ - min_)) * 255.).clamp(0., 255.) as u8)
                .collect::<Vec<_>>();

            let luma = Ops::resize_luma8_u8(
                &v,
                width as _,
                height as _,
                w1 as _,
                h1 as _,
                false,
                "Bilinear",
            )?;
            ys.push(Y::default().with_masks(&[Mask::new(&luma, w1, h1)?]));
        }

        Ok(ys)
    }
}
