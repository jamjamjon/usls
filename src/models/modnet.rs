use anyhow::Result;
use image::DynamicImage;
use ndarray::Axis;

use crate::{Mask, MinOptMax, Ops, Options, OrtEngine, X, Y};

#[derive(Debug)]
pub struct MODNet {
    engine: OrtEngine,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
}

impl MODNet {
    pub fn new(options: Options) -> Result<Self> {
        let mut engine = OrtEngine::new(&options)?;
        let (batch, height, width) = (
            engine.batch().to_owned(),
            engine.height().to_owned(),
            engine.width().to_owned(),
        );
        engine.dry_run()?;

        Ok(Self {
            engine,
            height,
            width,
            batch,
        })
    }

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<Y>> {
        let xs_ = X::apply(&[
            Ops::Resize(
                xs,
                self.height.opt as u32,
                self.width.opt as u32,
                "Lanczos3",
            ),
            Ops::Normalize(0., 255.),
            Ops::Nhwc2nchw,
        ])?;

        let ys = self.engine.run(vec![xs_])?;
        self.postprocess(ys, xs)
    }

    pub fn postprocess(&self, xs: Vec<X>, xs0: &[DynamicImage]) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        for (idx, luma) in xs[0].axis_iter(Axis(0)).enumerate() {
            let luma = luma
                .into_shape((self.height() as usize, self.width() as usize, 1))?
                .into_owned();
            let v = luma
                .into_raw_vec()
                .iter()
                .map(|x| (x * 255.0) as u8)
                .collect::<Vec<_>>();
            let luma: image::ImageBuffer<image::Luma<_>, Vec<u8>> =
                image::ImageBuffer::from_raw(self.width() as u32, self.height() as u32, v)
                    .expect("Faild to create image from ndarray");
            let luma = image::DynamicImage::from(luma);
            let luma = luma.resize_exact(
                xs0[idx].width(),
                xs0[idx].height(),
                image::imageops::FilterType::CatmullRom,
            );
            ys.push(Y::default().with_masks(&[Mask::default().with_mask(luma)]));
        }
        Ok(ys)
    }

    pub fn batch(&self) -> isize {
        self.batch.opt
    }

    pub fn width(&self) -> isize {
        self.width.opt
    }

    pub fn height(&self) -> isize {
        self.height.opt
    }
}
