use anyhow::Result;
use ort::tensor::TensorElementType;

use crate::{
    elapsed_module, inputs, Config, Device, Engine, Engines, FromConfig, Image, ImageProcessor,
    Model, Module, Scale, Task, Version, X,
};

pub type BaseModelVisual = BaseImageModel;

/// BaseImageModel - A simple image model wrapper implementing the Model trait.
///
/// This model provides a basic image processing pipeline with a single engine.
/// It's designed for simple image inference tasks like feature extraction
/// or encoding that don't require complex multi-engine architectures.
#[derive(aksr::Builder, Debug)]
pub struct BaseImageModel {
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

impl BaseImageModel {
    /// Complete encoding pipeline: preprocess + inference
    pub fn encode(&mut self, engines: &mut Engines, xs: &[Image]) -> Result<X> {
        let xs = elapsed_module!("BaseImageModel", "preprocess", self.processor.process(xs)?);
        let y = elapsed_module!(
            "BaseImageModel",
            "inference",
            engines
                .run(&Module::Model, inputs![xs]?)?
                .get::<f32>(0)
                .ok_or_else(|| anyhow::anyhow!("Failed to get output"))?
                .to_owned()
        );

        Ok(y)
    }
}

impl Model for BaseImageModel {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
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
            .inputs
            .dtypes
            .first()
            .copied()
            .unwrap_or(TensorElementType::Float32);
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let task = config.task;
        let scale = config.scale;
        let name = config.name;
        let version = config.version;

        let model = Self {
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
        };

        Ok((model, engine.into()))
    }
}
