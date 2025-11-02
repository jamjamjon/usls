//! Pre-built models for various vision and vision-language tasks.
//!
//! This module provides ready-to-use implementations of state-of-the-art models including:
//! - **Object Detection**: YOLO series, RT-DETR, RF-DETR, GroundingDINO, OWLv2
//! - **Instance Segmentation**: YOLO, SAM, SAM2, FastSAM
//! - **Keypoint Detection**: RTMPose, DWPose, RTMO, RTMW
//! - **Text Processing**: SVTR, TrOCR, DB, SLANet
//! - **Vision-Language**: CLIP, BLIP, Florence2, Moondream2, FastVLM, SmolVLM
//! - **Other Tasks**: Depth estimation, image matting, super-resolution, background removal
//!
//! Each model is available as a separate feature flag for minimal dependencies.

#[cfg(feature = "yolo")]
mod yolo;

#[cfg(feature = "sam")]
mod sam;

#[cfg(feature = "clip")]
mod clip;

#[cfg(feature = "apisr")]
mod apisr;

#[cfg(feature = "image-classifier")]
mod beit;

#[cfg(feature = "ben2")]
mod ben2;

#[cfg(feature = "blip")]
mod blip;

#[cfg(feature = "image-classifier")]
mod convnext;

#[cfg(feature = "db")]
mod db;
#[cfg(feature = "db")]
mod fast;
#[cfg(feature = "db")]
mod linknet;

#[cfg(feature = "image-classifier")]
mod deit;

#[cfg(feature = "depth-anything")]
mod depth_anything;

#[cfg(feature = "depth-pro")]
mod depth_pro;

#[cfg(feature = "dino")]
mod dinov2;

#[cfg(feature = "rtmpose")]
mod dwpose;

#[cfg(feature = "image-classifier")]
mod fastvit;

#[cfg(feature = "fastvlm")]
mod fastvlm;

#[cfg(feature = "florence2")]
mod florence2;

#[cfg(feature = "grounding-dino")]
mod grounding_dino;

#[cfg(feature = "mediapipe-segmenter")]
mod mediapipe_segmenter;

#[cfg(feature = "image-classifier")]
mod mobileone;

#[cfg(feature = "modnet")]
mod modnet;

#[cfg(feature = "moondream2")]
mod moondream2;

#[cfg(feature = "owl")]
mod owl;

#[cfg(feature = "picodet")]
mod picodet;

#[cfg(feature = "pipeline")]
mod pipeline;

#[cfg(feature = "rfdetr")]
mod rfdetr;

#[cfg(feature = "rmbg")]
mod rmbg;

#[cfg(feature = "rtdetr")]
mod rtdetr;

#[cfg(feature = "rtmo")]
mod rtmo;

#[cfg(feature = "rtmpose")]
mod rtmpose;
#[cfg(feature = "rtmpose")]
mod rtmw;

#[cfg(feature = "sapiens")]
mod sapiens;

#[cfg(all(feature = "slanet", feature = "pipeline"))]
mod slanet;

#[cfg(feature = "smolvlm")]
mod smolvlm;

#[cfg(feature = "svtr")]
mod svtr;

#[cfg(feature = "swin2sr")]
mod swin2sr;

#[cfg(feature = "trocr")]
mod trocr;

#[cfg(feature = "yolop")]
mod yolop;

#[cfg(feature = "yoloe")]
mod yoloe;

#[cfg(feature = "sam")]
mod sam2;

#[cfg(feature = "rtdetr")]
mod d_fine;

#[cfg(feature = "dino")]
mod dinov3;

#[cfg(feature = "rtdetr")]
mod deim;

#[cfg(feature = "rtdetr")]
mod deimv2;

#[cfg(feature = "yolo")]
pub use yolo::*;

#[cfg(feature = "sam")]
pub use sam::*;

#[cfg(feature = "clip")]
pub use clip::*;

#[cfg(feature = "blip")]
pub use blip::*;

#[cfg(feature = "db")]
pub use db::*;

#[cfg(feature = "depth-anything")]
pub use depth_anything::*;

#[cfg(feature = "depth-pro")]
pub use depth_pro::*;

#[cfg(feature = "dino")]
pub use dinov2::*;

#[cfg(feature = "fastvlm")]
pub use fastvlm::*;

#[cfg(feature = "florence2")]
pub use florence2::*;

#[cfg(feature = "grounding-dino")]
pub use grounding_dino::*;

#[cfg(feature = "mediapipe-segmenter")]
pub use mediapipe_segmenter::*;

#[cfg(feature = "modnet")]
pub use modnet::*;

#[cfg(feature = "moondream2")]
pub use moondream2::*;

#[cfg(feature = "owl")]
pub use owl::*;

#[cfg(feature = "picodet")]
pub use picodet::*;

#[cfg(feature = "pipeline")]
pub use pipeline::*;

#[cfg(feature = "rfdetr")]
pub use rfdetr::*;

#[cfg(feature = "rmbg")]
pub use rmbg::*;

#[cfg(feature = "rtdetr")]
pub use rtdetr::*;

#[cfg(feature = "rtmo")]
pub use rtmo::*;

#[cfg(feature = "rtmpose")]
pub use rtmpose::*;

#[cfg(feature = "sapiens")]
pub use sapiens::*;

#[cfg(feature = "slanet")]
pub use slanet::*;

#[cfg(feature = "smolvlm")]
pub use smolvlm::*;

#[cfg(feature = "svtr")]
pub use svtr::*;

#[cfg(feature = "swin2sr")]
pub use swin2sr::*;

#[cfg(feature = "trocr")]
pub use trocr::*;

#[cfg(feature = "yolop")]
pub use yolop::*;

#[cfg(feature = "sam")]
pub use sam2::*;

#[cfg(feature = "rtdetr")]
pub use d_fine::*;

#[cfg(feature = "dino")]
pub use dinov3::*;

#[cfg(feature = "rtdetr")]
pub use deim::*;

#[cfg(feature = "rtdetr")]
pub use deimv2::*;
