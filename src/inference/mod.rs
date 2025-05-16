#[cfg(any(feature = "ort-download-binaries", feature = "ort-load-dynamic"))]
mod engine;
mod engine_config;
mod hbb;
mod image;
mod instance_meta;
mod keypoint;
mod mask;
mod model_config;
mod obb;
mod polygon;
mod prob;
mod skeleton;
mod x;
mod xs;
mod y;
#[cfg(any(feature = "ort-download-binaries", feature = "ort-load-dynamic"))]
#[allow(clippy::all)]
pub(crate) mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[cfg(any(feature = "ort-download-binaries", feature = "ort-load-dynamic"))]
pub use engine::*;
pub use engine_config::EngineConfig;
pub use hbb::*;
pub use image::*;
pub use instance_meta::*;
pub use keypoint::*;
pub use mask::*;
pub use model_config::*;
pub use obb::*;
pub use polygon::*;
pub use prob::*;
pub use skeleton::*;
pub use x::X;
pub use xs::Xs;
pub use y::*;
