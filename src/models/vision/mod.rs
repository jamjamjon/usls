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
pub mod pipeline;

// Classification (config-only, used with ImageClassifier pipeline)
mod beit;
mod convnext;
mod deit;
mod fastvit;
mod mobileone;

// Object Detection
pub mod d_fine;
pub mod deim;
pub mod deimv2;
pub mod picodet;
pub mod rfdetr;
pub mod rtdetr;
pub mod yolo;
pub mod yolop;

// Segmentation
pub mod mediapipe_segmenter;
pub mod sam;
pub mod sam2;
pub mod sapiens;

// Pose Estimation
mod dwpose; // config-only
pub mod rtmo;
pub mod rtmpose;
mod rtmw; // config-only

// Depth Estimation
pub mod depth_anything;
pub mod depth_pro;

// Feature Extraction
pub mod dinov2;
pub mod dinov3;

// Recognition
pub mod ram;

// Text Detection
pub mod db;
mod fast; // config-only
mod linknet; // config-only

// Text Recognition
pub mod svtr;

// Background Removal / Matting
mod ben2; // config-only
pub mod modnet;
pub mod rmbg;

// Super Resolution
mod apisr;
pub mod swin2sr; // config-only

// Table Structure Recognition
pub mod slanet;

// Re-exports
pub use d_fine::*;
pub use db::*;
pub use deim::*;
pub use deimv2::*;
pub use depth_anything::*;
pub use depth_pro::*;
pub use dinov2::*;
pub use dinov3::*;
pub use mediapipe_segmenter::*;
pub use modnet::*;
pub use picodet::*;
pub use pipeline::*;
pub use ram::*;
pub use rfdetr::*;
pub use rmbg::*;
pub use rtdetr::*;
pub use rtmo::*;
pub use rtmpose::*;
pub use sam::*;
pub use sam2::*;
pub use sapiens::*;
pub use slanet::*;
pub use svtr::*;
pub use swin2sr::*;
pub use yolo::*;
pub use yolop::*;
