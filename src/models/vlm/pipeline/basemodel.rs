use aksr::Builder;
use anyhow::Result;

use crate::{
    ort_inputs, Config, DType, Device, Engine, FromConfig, ImageProcessor, Scale, Task,
    TextProcessor, Version, X,
};

#[derive(Debug, Builder)]
pub struct BaseModelTextual {
    engine: Engine,
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

impl BaseModelTextual {
    pub fn new(config: Config) -> Result<Self> {
        let device = config
            .get_module(&crate::Module::Model)
            .map(|e| e.device)
            .unwrap_or(Device::Cpu(0));
        let dtype = config
            .get_module(&crate::Module::Model)
            .map(|e| e.dtype)
            .unwrap_or(crate::DType::Fp32);
        let engine_config = config
            .get_module(&crate::Module::Model)
            .ok_or_else(|| anyhow::anyhow!("Model engine not configured"))?
            .clone();

        let task = config.task;
        let scale = config.scale;
        let name = config.name;
        let version = config.version;

        let image_processor = ImageProcessor::from_config(config.image_processor)?;
        #[cfg(feature = "vlm")]
        let text_processor = TextProcessor::from_config(config.text_processor)?;
        #[cfg(not(feature = "vlm"))]
        let text_processor = TextProcessor::default();

        let engine = Engine::from_config(engine_config)?;
        let (batch, spec) = (engine.batch().opt(), engine.spec().to_owned());

        Ok(Self {
            engine,
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
        })
    }

    pub fn inference(&mut self, xs: X) -> Result<X> {
        let output = self.engine.run(ort_inputs![xs]?)?;
        Ok(X::from(output.get::<f32>(0)?))
    }
}
