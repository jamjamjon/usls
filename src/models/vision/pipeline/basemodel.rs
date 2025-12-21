use anyhow::Result;
use ort::tensor::TensorElementType;

use crate::{
    elapsed_module, ort_inputs, Config, Device, Engine, FromConfig, Image, ImageProcessor, Module,
    Scale, Task, Version, X,
};

#[derive(Debug)]
#[allow(dead_code)]
pub struct BaseModelVisual {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    processor: ImageProcessor,
    spec: String,
    name: &'static str,
    device: Device,
    dtype: TensorElementType,
    task: Option<Task>,
    scale: Option<Scale>,
    version: Option<Version>,
}

impl BaseModelVisual {
    pub fn new(mut config: Config) -> Result<Self> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let err_msg = "You need to specify the image height and image width for visual model.";
        let (batch, height, width, spec) = (
            engine.batch().opt(),
            engine.try_height().expect(err_msg).opt(),
            engine.try_width().expect(err_msg).opt(),
            engine.spec().to_owned(),
        );
        let device = *engine.device();
        let dtype = engine
            .onnx
            .as_ref()
            .map(|x| x.inputs.dtypes[0])
            .unwrap_or(TensorElementType::Float32);
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let task = config.task;
        let scale = config.scale;
        let name = config.name;
        let version = config.version;

        Ok(Self {
            engine,
            height,
            width,
            batch,
            processor,
            spec,
            name,
            device,
            dtype,
            task,
            scale,
            version,
        })
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn processor(&self) -> &ImageProcessor {
        &self.processor
    }

    pub fn preprocess(&mut self, xs: &[Image]) -> Result<X> {
        let processed = self.processor.process(xs)?;
        self.batch = xs.len(); // update
        processed.as_host()
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    pub fn inference(&mut self, xs: X) -> Result<X> {
        let output = self.engine.run(ort_inputs![xs]?)?;
        Ok(X::from(output.get::<f32>(0)?))
    }

    pub fn encode(&mut self, xs: &[Image]) -> Result<X> {
        let xs = elapsed_module!("BaseModelVisual", "visual-preprocess", self.preprocess(xs)?);
        let xs = elapsed_module!("BaseModelVisual", "visual-inference", self.inference(xs)?);

        Ok(xs)
    }
}
