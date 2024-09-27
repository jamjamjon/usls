use anyhow::Result;
use image::DynamicImage;
use ndarray::Axis;

use crate::{Mask, MinOptMax, Ops, Options, OrtEngine, Xs, X, Y};

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
                self.height.opt() as u32,
                self.width.opt() as u32,
                "Lanczos3",
            ),
            Ops::Normalize(0., 255.),
            Ops::Nhwc2nchw,
        ])?;

        let ys = self.engine.run(Xs::from(xs_))?;
        self.postprocess(ys, xs)
    }

    pub fn postprocess(&self, xs: Xs, xs0: &[DynamicImage]) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        for (idx, luma) in xs[0].axis_iter(Axis(0)).enumerate() {
            let (w1, h1) = (xs0[idx].width(), xs0[idx].height());
            let luma = luma.mapv(|x| (x * 255.0) as u8);
            let luma = Ops::resize_luma8_u8(
                &luma.into_raw_vec_and_offset().0,
                self.width() as _,
                self.height() as _,
                w1 as _,
                h1 as _,
                false,
                "Bilinear",
            )?;
            let luma: image::ImageBuffer<image::Luma<_>, Vec<_>> =
                match image::ImageBuffer::from_raw(w1 as _, h1 as _, luma) {
                    None => continue,
                    Some(x) => x,
                };
            ys.push(Y::default().with_masks(&[Mask::default().with_mask(luma)]));
        }
        Ok(ys)
    }

    pub fn batch(&self) -> isize {
        self.batch.opt() as _
    }

    pub fn width(&self) -> isize {
        self.width.opt() as _
    }

    pub fn height(&self) -> isize {
        self.height.opt() as _
    }
}
