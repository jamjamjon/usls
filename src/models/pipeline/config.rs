use crate::{impl_model_config_methods, impl_process_config_methods};

#[derive(aksr::Builder, Debug, Clone)]
pub struct BaseConfig {
    pub model: crate::ModelConfig,
    pub processor: crate::ProcessorConfig,
}

impl Default for BaseConfig {
    fn default() -> Self {
        Self {
            model: crate::ModelConfig::default()
                .with_name("db")
                .with_ixx(0, 0, (1, 1, 8).into())
                .with_ixx(0, 1, 3.into())
                .with_ixx(0, 2, 640.into())
                .with_ixx(0, 3, 640.into()),
            processor: crate::ProcessorConfig::default(),
        }
    }
}

impl_model_config_methods!(BaseConfig, model);
impl_process_config_methods!(BaseConfig, processor);
