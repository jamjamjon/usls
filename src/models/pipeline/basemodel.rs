use aksr::Builder;
use anyhow::Result;

use crate::{
    elapsed, Config, DType, Device, Engine, Image, Processor, Scale, Task, Ts, Version, Xs, X,
};

#[derive(Debug, Builder)]
pub struct BaseModelVisual {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    processor: Processor,
    ts: Ts,
    spec: String,
    name: &'static str,
    device: Device,
    dtype: DType,
    task: Option<Task>,
    scale: Option<Scale>,
    version: Option<Version>,
}

impl BaseModelVisual {
    pub fn summary(&self) {
        self.ts.summary();
    }

    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let err_msg = "You need to specify the image height and image width for visual model.";
        let (batch, height, width, ts, spec) = (
            engine.batch().opt(),
            engine.try_height().expect(err_msg).opt(),
            engine.try_width().expect(err_msg).opt(),
            engine.ts.clone(),
            engine.spec().to_owned(),
        );
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let device = config.model.device;
        let task = config.task;
        let scale = config.scale;
        let dtype = config.model.dtype;
        let name = config.name;
        let version = config.version;

        Ok(Self {
            engine,
            height,
            width,
            batch,
            processor,
            ts,
            spec,
            dtype,
            task,
            scale,
            device,
            version,
            name,
        })
    }

    pub fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        let x = self.processor.process_images(xs)?;
        self.batch = xs.len(); // update

        Ok(x.into())
    }

    pub fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn encode(&mut self, xs: &[Image]) -> Result<X> {
        let xs = elapsed!("visual-preprocess", self.ts, { self.preprocess(xs)? });
        let xs = elapsed!("visual-inference", self.ts, { self.inference(xs)? });

        Ok(xs[0].to_owned())
    }
}

#[derive(Debug, Builder)]
pub struct BaseModelTextual {
    engine: Engine,
    batch: usize,
    processor: Processor,
    ts: Ts,
    spec: String,
    name: &'static str,
    device: Device,
    dtype: DType,
    task: Option<Task>,
    scale: Option<Scale>,
    version: Option<Version>,
}

impl BaseModelTextual {
    pub fn summary(&self) {
        self.ts.summary();
    }

    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let (batch, ts, spec) = (
            engine.batch().opt(),
            engine.ts.clone(),
            engine.spec().to_owned(),
        );
        let processor = Processor::try_from_config(&config.processor)?;
        let device = config.model.device;
        let dtype = config.model.dtype;
        let task = config.task;
        let scale = config.scale;
        let name = config.name;
        let version = config.version;

        Ok(Self {
            engine,
            batch,
            processor,
            ts,
            spec,
            dtype,
            task,
            scale,
            device,
            version,
            name,
        })
    }

    pub fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }
}
