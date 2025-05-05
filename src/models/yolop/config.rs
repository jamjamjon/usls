/// Model configuration for `YOLOP`
impl crate::Options {
    pub fn yolop() -> Self {
        Self::default()
            .with_model_name("yolop")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 2, 640.into())
            .with_model_ixx(0, 3, 640.into())
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_resize_filter("Bilinear")
            .with_normalize(true)
            .with_class_confs(&[0.3])
    }

    pub fn yolop_v2_480x800() -> Self {
        Self::yolop().with_model_file("v2-480x800.onnx")
    }

    pub fn yolop_v2_736x1280() -> Self {
        Self::yolop().with_model_file("v2-736x1280.onnx")
    }
}

#[derive(aksr::Builder, Debug, Clone)]
pub struct YOLOPConfig {
    pub model: crate::ModelConfig,
    pub processor: crate::ProcessorConfig,
    pub class_names: Vec<String>,
    pub class_confs: Vec<f32>,
    pub iou: f32,
}

impl Default for YOLOPConfig {
    fn default() -> Self {
        Self {
            model: crate::ModelConfig::default()
                .with_name("yolop")
                .with_ixx(0, 0, 1.into())
                .with_ixx(0, 1, 3.into())
                .with_ixx(0, 2, 512.into())
                .with_ixx(0, 3, 512.into()),
            processor: crate::ProcessorConfig::default()
                .with_resize_mode(crate::ResizeMode::FitAdaptive)
                .with_resize_filter("Bilinear"),
            class_confs: vec![0.2f32],
            class_names: vec![],
            iou: 0.45f32,
        }
    }
}

use crate::{impl_model_config_methods, impl_process_config_methods};

impl_model_config_methods!(YOLOPConfig, model);
impl_process_config_methods!(YOLOPConfig, processor);
