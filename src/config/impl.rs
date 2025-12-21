use std::collections::HashMap;

use aksr::Builder;

#[cfg(feature = "vlm")]
use crate::TextProcessorConfig;
use crate::{ImageProcessorConfig, InferenceParams, Module, ORTConfig, Scale, Task, Version};

/// Configuration for model inference including modules, processors, and task settings.
///
/// # Architecture Changes
/// - **Modules**: Now stored in `HashMap<Module, ORTConfig>` for dynamic management
/// - **Inference Params**: Consolidated into `InferenceParams` struct
/// - **Move Semantics**: Use `take_module()` to consume modules with zero-copy
#[derive(Builder, Debug, Clone, Default)]
pub struct Config {
    // Basics
    pub name: &'static str,
    pub version: Option<Version>,
    pub task: Option<Task>,
    pub scale: Option<Scale>,

    // Modules (dynamic registry)
    pub modules: HashMap<Module, ORTConfig>,

    // Processors
    pub image_processor: ImageProcessorConfig,
    #[cfg(feature = "vlm")]
    pub text_processor: TextProcessorConfig,

    // Inference Parameters
    pub inference: InferenceParams,
}

impl Config {
    /// Commit all module configurations (download models, resolve paths, etc.).
    pub fn commit(mut self) -> anyhow::Result<Self> {
        // Special case for YOLO: generate file name from version/scale/task
        if self.name == "yolo" {
            if let Some(model_config) = self.modules.get_mut(&Module::Model) {
                if model_config.file.is_empty() {
                    let mut y = String::new();
                    if let Some(x) = self.version {
                        y.push_str(&x.to_string());
                    }
                    if let Some(ref x) = self.scale {
                        y.push_str(&format!("-{}", x));
                    }
                    if let Some(ref x) = self.task {
                        y.push_str(&format!("-{}", x.yolo_str()));
                    }
                    y.push_str(".onnx");
                    model_config.file = y;
                }
            }
        }

        // Commit each module config
        let name = self.name;
        for (_, config) in self.modules.iter_mut() {
            if !config.file.is_empty() {
                *config = std::mem::take(config).try_commit(name)?;
            }
        }

        Ok(self)
    }
}
