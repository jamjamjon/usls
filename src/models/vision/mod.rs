//! Pure vision models that process images without text understanding.
//!
//! All models are compiled when the `vision` feature is enabled.
//!
//! # Model Categories
//!
//! - **Classification**: `beit`, `convnext`, `deit`, `fastvit`, `mobileone` (config-only, use with `ImageClassifier`)
//! - **Detection**: `yolo`, `yolop`, `rtdetr`, `rfdetr`, `picodet`, `d_fine`, `deim`, `deimv2`
//! - **Segmentation**: `sam`, `sam2`, `mediapipe_segmenter`, `sapiens`
//! - **Pose**: `rtmpose`, `rtmw`, `dwpose`, `rtmo`
//! - **Depth**: `depth_anything`, `depth_pro`
//! - **Feature**: `dinov2`, `dinov3`, `clip`, `blip`, `ram`
//! - **OCR**: `db`, `fast`, `linknet`, `svtr`, `slanet`
//! - **Matting**: `rmbg`, `ben2`, `modnet`
//! - **Super-resolution**: `swin2sr`, `apisr`

// Pipeline
mod pipeline;

// Classification
mod beit;
mod convnext;
mod deit;
mod fastvit;
mod mobileone;

// Object Detection
mod d_fine;
mod deim;
mod deimv2;
mod picodet;
mod rfdetr;
mod rtdetr;
mod yolo;
mod yolop;

// Segmentation
mod fastsam;
mod mediapipe_segmenter;
mod sam;
mod sam2;
mod sam3_tracker;
mod sapiens;
mod yoloe_prompt_free;

// Pose Estimation
mod dwpose;
mod rtmo;
mod rtmpose;
mod rtmw;

// Depth Estimation
mod depth_anything;
mod depth_pro;

// Feature Extraction
mod dinov2;
mod dinov3;

// Recognition
mod ram;

// Text Detection
mod db;
mod fast;
mod linknet;

// Text Recognition
mod svtr;

// Background Removal / Matting
mod ben2;
mod modnet;
mod rmbg;

// Super Resolution
mod apisr;
mod swin2sr;

// Table Structure Recognition
mod slanet;

// Re-exports
pub use apisr::*;
pub use beit::*;
pub use ben2::*;
pub use convnext::*;
pub use d_fine::*;
pub use db::*;
pub use deim::*;
pub use deimv2::*;
pub use deit::*;
pub use depth_anything::*;
pub use depth_pro::*;
pub use dinov2::*;
pub use dinov3::*;
pub use dwpose::*;
pub use fast::*;
pub use fastsam::*;
pub use fastvit::*;
pub use linknet::*;
pub use mediapipe_segmenter::*;
pub use mobileone::*;
pub use modnet::*;
pub use picodet::*;
pub use pipeline::*;
pub use ram::*;
pub use rfdetr::*;
pub use rmbg::*;
pub use rtdetr::*;
pub use rtmo::*;
pub use rtmpose::*;
pub use rtmw::*;
pub use sam::*;
pub use sam2::*;
pub use sam3_tracker::*;
pub use sapiens::*;
pub use slanet::*;
pub use svtr::*;
pub use swin2sr::*;
pub use yolo::*;
pub use yoloe_prompt_free::*;
pub use yolop::*;
