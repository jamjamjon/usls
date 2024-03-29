use crate::{ops, MinOptMax, Options, OrtEngine};
use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array, IxDyn};

#[derive(Debug)]
pub struct Dinov2 {
    engine: OrtEngine,
    pub height: MinOptMax,
    pub width: MinOptMax,
    pub batch: MinOptMax,
}

impl Dinov2 {
    pub fn new(options: &Options) -> Result<Self> {
        let engine = OrtEngine::new(options)?;
        let (batch, height, width) = (
            engine.inputs_minoptmax()[0][0].to_owned(),
            engine.inputs_minoptmax()[0][2].to_owned(),
            engine.inputs_minoptmax()[0][3].to_owned(),
        );
        engine.dry_run()?;

        Ok(Self {
            engine,
            height,
            width,
            batch,
        })
    }

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Array<f32, IxDyn>> {
        let xs_ = ops::resize(xs, self.height.opt as u32, self.width.opt as u32, true)?;
        let ys: Vec<Array<f32, IxDyn>> = self.engine.run(&[xs_])?;
        let ys = ys[0].to_owned();
        let ys = ops::norm(&ys);
        Ok(ys)
    }
}
