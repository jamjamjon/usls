use crate::{Mask, MinOptMax, Ops, Options, OrtEngine, Xs, X, Y};
use anyhow::Result;
use image::DynamicImage;
use ndarray::Axis;

#[derive(Debug)]
pub struct DepthPro {
    engine: OrtEngine,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
}

impl DepthPro {
    pub fn new(options: Options) -> Result<Self> {
        let mut engine = OrtEngine::new(&options)?;
        let (batch, height, width) = (
            engine.batch().clone(),
            engine.height().clone(),
            engine.width().clone(),
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
                "Bilinear",
            ),
            Ops::Normalize(0., 255.),
            Ops::Standardize(&[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5], 3),
            Ops::Nhwc2nchw,
        ])?;
        let ys = self.engine.run(Xs::from(xs_))?;

        self.postprocess(ys, xs)
    }

    pub fn postprocess(&self, xs: Xs, xs0: &[DynamicImage]) -> Result<Vec<Y>> {
        let (predicted_depth, _focallength_px) = (&xs["predicted_depth"], &xs["focallength_px"]);
        let predicted_depth = predicted_depth.mapv(|x| 1. / x);

        let mut ys: Vec<Y> = Vec::new();
        for (idx, luma) in predicted_depth.axis_iter(Axis(0)).enumerate() {
            let (w1, h1) = (xs0[idx].width(), xs0[idx].height());
            let v = luma.into_owned().into_raw_vec_and_offset().0;
            let max_ = v.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
            let min_ = v.iter().min_by(|x, y| x.total_cmp(y)).unwrap();
            let v = v
                .iter()
                .map(|x| (((*x - min_) / (max_ - min_)) * 255.).clamp(0., 255.) as u8)
                .collect::<Vec<_>>();

            let luma = Ops::resize_luma8_u8(
                &v,
                self.width.opt() as _,
                self.height.opt() as _,
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
}
