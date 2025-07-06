use aksr::Builder;
use anyhow::Result;

use crate::{elapsed_module, Config, Engine, Image, Mask, Ops, Processor, Xs, Y};

#[derive(Debug, Builder)]
pub struct DepthAnything {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    spec: String,
    processor: Processor,
}

impl DepthAnything {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let spec = engine.spec().to_string();

        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&518.into()).opt(),
            engine.try_width().unwrap_or(&518.into()).opt(),
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

    fn postprocess(&mut self, xs: Xs) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        for (idx, luma) in xs[0].axis_iter(ndarray::Axis(0)).enumerate() {
            // image size
            let (h1, w1) = (
                self.processor.images_transform_info[idx].height_src,
                self.processor.images_transform_info[idx].width_src,
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
                self.width() as _,
                self.height() as _,
                w1 as _,
                h1 as _,
                false,
                "Bilinear",
            )?;
            ys.push(Y::default().with_masks(&[Mask::new(&luma, w1, h1)?]));
        }

        Ok(ys)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("DepthAnything", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("DepthAnything", "inference", self.inference(ys)?);
        let ys = elapsed_module!("DepthAnything", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }
}
