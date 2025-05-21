#[cfg(any(feature = "ort-download-binaries", feature = "ort-load-dynamic"))]
mod engine;
mod hbb;
mod image;
mod instance_meta;
mod keypoint;
mod mask;
mod obb;
mod polygon;
mod prob;
mod skeleton;
mod text;
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
pub use hbb::*;
pub use image::*;
pub use instance_meta::*;
pub use keypoint::*;
pub use mask::*;
pub use obb::*;
pub use polygon::*;
pub use prob::*;
pub use skeleton::*;
pub use text::*;
pub use x::X;
pub use xs::Xs;
pub use y::*;
