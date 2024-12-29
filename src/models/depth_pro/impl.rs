use aksr::Builder;
use anyhow::Result;
use image::DynamicImage;
use ndarray::Axis;

use crate::{elapsed, Engine, Mask, Ops, Options, Processor, Ts, Xs, Ys, Y};

#[derive(Builder, Debug)]
pub struct DepthPro {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    ts: Ts,
    spec: String,
    processor: Processor,
}

impl DepthPro {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let spec = engine.spec().to_string();
        let (batch, height, width, ts) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&512.into()).opt(),
            engine.try_width().unwrap_or(&512.into()).opt(),
            engine.ts().clone(),
        );
        let processor = options
            .to_processor()?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            engine,
            height,
            width,
            batch,
            ts,
            spec,
            processor,
        })
    }

    fn preprocess(&mut self, xs: &[DynamicImage]) -> Result<Xs> {
        Ok(self.processor.process_images(xs)?.into())
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    fn postprocess(&mut self, xs: Xs) -> Result<Ys> {
        let (predicted_depth, _focallength_px) = (&xs["predicted_depth"], &xs["focallength_px"]);
        let predicted_depth = predicted_depth.mapv(|x| 1. / x);

        let mut ys: Vec<Y> = Vec::new();
        for (idx, luma) in predicted_depth.axis_iter(Axis(0)).enumerate() {
            let (h1, w1) = self.processor.image0s_size[idx];
            let v = luma.into_owned().into_raw_vec_and_offset().0;
            let max_ = v.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
            let min_ = v.iter().min_by(|x, y| x.total_cmp(y)).unwrap();
            let v = v
                .iter()
                .map(|x| (((*x - min_) / (max_ - min_)) * 255.).clamp(0., 255.) as u8)
                .collect::<Vec<_>>();

            let luma = Ops::resize_luma8_u8(
                &v,
                self.width as _,
                self.height as _,
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

        Ok(ys.into())
    }

    pub fn forward(&mut self, xs: &[DynamicImage]) -> Result<Ys> {
        let ys = elapsed!("preprocess", self.ts, { self.preprocess(xs)? });
        let ys = elapsed!("inference", self.ts, { self.inference(ys)? });
        let ys = elapsed!("postprocess", self.ts, { self.postprocess(ys)? });

        Ok(ys)
    }

    pub fn summary(&mut self) {
        self.ts.summary();
    }
}
