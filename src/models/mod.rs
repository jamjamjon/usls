mod beit;
mod blip;
mod clip;
mod convnext;
mod d_fine;
mod db;
mod deim;
mod deit;
mod depth_anything;
mod depth_pro;
mod dinov2;
mod fast;
mod fastvit;
mod florence2;
mod grounding_dino;
mod linknet;
mod mobileone;
mod modnet;
mod moondream2;
mod owl;
mod picodet;
mod pipeline;
mod rfdetr;
mod rtdetr;
mod rtmo;
mod sam;
mod sapiens;
mod slanet;
mod smolvlm;
mod svtr;
mod trocr;
mod yolo;
mod yolop;

pub use blip::*;
pub use clip::*;
pub use db::*;
pub use depth_anything::*;
pub use depth_pro::*;
pub use dinov2::*;
pub use florence2::*;
pub use grounding_dino::*;
pub use modnet::*;
pub use moondream2::*;
pub use owl::*;
pub use picodet::*;
pub use pipeline::*;
pub use rfdetr::*;
pub use rtdetr::*;
pub use rtmo::*;
pub use sam::*;
pub use sapiens::*;
pub use slanet::*;
pub use smolvlm::*;
pub use svtr::*;
pub use trocr::*;
pub use yolo::*;
pub use yolop::*;

#[derive(aksr::Builder, Debug, Clone)]
pub struct ObjectDetectionConfig {
    pub model: crate::ModelConfig,
    pub processor: crate::ProcessorConfig,
    pub class_names: Vec<String>,
    pub text_names: Vec<String>,
    pub class_confs: Vec<f32>,
    pub text_confs: Vec<f32>,
    pub apply_nms: bool,
}

impl Default for ObjectDetectionConfig {
    fn default() -> Self {
        Self {
            model: crate::ModelConfig::default()
                .with_ixx(0, 0, 1.into())
                .with_ixx(0, 1, 3.into())
                .with_ixx(0, 2, 640.into())
                .with_ixx(0, 3, 640.into()),
            processor: crate::ProcessorConfig::default()
                .with_resize_mode(crate::ResizeMode::FitAdaptive)
                .with_resize_filter("CatmullRom"),
            class_confs: vec![0.2f32],
            class_names: vec![],
            text_names: vec![],
            text_confs: vec![0.2f32],
            apply_nms: false,
        }
    }
}

use crate::{impl_model_config_methods, impl_process_config_methods};

impl_model_config_methods!(ObjectDetectionConfig, model);
impl_process_config_methods!(ObjectDetectionConfig, processor);

#[derive(aksr::Builder, Debug, Clone)]
pub struct KeypointDetectionConfig {
    pub model: crate::ModelConfig,
    pub processor: crate::ProcessorConfig,
    pub class_names: Vec<String>,
    pub class_confs: Vec<f32>,
    pub keypoint_confs: Vec<f32>,
    pub keypoint_names: Vec<String>,
    #[args(aka = "nk")]
    pub num_keypoints: Option<usize>,
    pub apply_nms: bool,
}

impl Default for KeypointDetectionConfig {
    fn default() -> Self {
        Self {
            model: crate::ModelConfig::default()
                .with_ixx(0, 0, 1.into())
                .with_ixx(0, 1, 3.into())
                .with_ixx(0, 2, 640.into())
                .with_ixx(0, 3, 640.into()),
            processor: crate::ProcessorConfig::default()
                .with_resize_mode(crate::ResizeMode::FitAdaptive)
                .with_resize_filter("CatmullRom"),
            class_confs: vec![0.2f32],
            class_names: vec![],
            keypoint_confs: vec![0.5f32],
            keypoint_names: vec![],
            num_keypoints: None,
            apply_nms: false,
        }
    }
}

impl_model_config_methods!(KeypointDetectionConfig, model);
impl_process_config_methods!(KeypointDetectionConfig, processor);
