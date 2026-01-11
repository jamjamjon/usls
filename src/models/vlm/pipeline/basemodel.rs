use aksr::Builder;
use anyhow::Result;

use crate::{
    inputs, Config, DType, Device, Engine, Engines, FromConfig, ImageProcessor, Model, Module,
    Scale, Task, TextProcessor, Version, X, Y,
};


pub type BaseModelTextual = BaseTextModel;

/// BaseTextModel - A simple text model wrapper implementing the Model trait.
///
/// This model provides a basic text processing pipeline with a single engine.
/// It's designed for simple text inference tasks that don't require complex
/// multi-engine architectures.
#[derive(Debug, Builder)]
pub struct BaseTextModel {
    batch: usize,
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
    spec: String,
    name: &'static str,
    device: Device,
    dtype: DType,
    task: Option<Task>,
    scale: Option<Scale>,
    version: Option<Version>,
}

impl Model for BaseTextModel {
    type Input<'a> = X;

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(config: Config) -> Result<(Self, Engines)> {
        let device = config
            .get_module(&Module::Model)
            .map(|e| e.device)
            .unwrap_or(Device::Cpu(0));
        let dtype = config
            .get_module(&Module::Model)
            .map(|e| e.dtype)
            .unwrap_or(crate::DType::Fp32);
        let engine_config = config
            .get_module(&Module::Model)
            .ok_or_else(|| anyhow::anyhow!("Model engine not configured"))?
            .clone();

        let task = config.task;
        let scale = config.scale;
        let name = config.name;
        let version = config.version;
        let image_processor = ImageProcessor::from_config(config.image_processor)?;
        let text_processor = TextProcessor::from_config(config.text_processor)?;
        let engine = Engine::from_config(engine_config)?;
        let (batch, spec) = (engine.batch().opt(), engine.spec().to_owned());

        let model = Self {
            batch,
            image_processor,
            text_processor,
            spec,
            dtype,
            task,
            scale,
            device,
            version,
            name,
        };

        Ok((model, engine.into()))
    }

    // fn run(&mut self, engines: &mut Engines, input: Self::Input<'_>) -> Result<Vec<Y>> {
    //     let output = self.inference(engines, input)?;
    //     Ok(vec![Y::default().with_embedding(output)])
    // }
}
