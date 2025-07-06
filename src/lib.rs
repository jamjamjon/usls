//! # usls
//!
//! `usls` is a cross-platform Rust library that provides efficient inference for SOTA vision and multi-modal models using ONNX Runtime (typically under 1B parameters).
//!
//! ## 📚 Documentation
//! - [API Documentation](https://docs.rs/usls/latest/usls/)
//! - [Examples](https://github.com/jamjamjon/usls/tree/main/examples)
//! ## 🚀 Quick Start
//!
//! ```bash
//! # CPU
//! cargo run -r --example yolo  # YOLOv8 detect by default
//!
//! # NVIDIA CUDA
//! cargo run -r -F cuda --example yolo -- --device cuda:0
//!
//! # NVIDIA TensorRT
//! cargo run -r -F tensorrt --example yolo -- --device tensorrt:0
//!
//! # Apple Silicon CoreML
//! cargo run -r -F coreml --example yolo -- --device coreml
//!
//! # Intel OpenVINO
//! cargo run -r -F openvino -F ort-load-dynamic --example yolo -- --device openvino:CPU
//! ```
//!
//! ## ⚡ Cargo Features
//! - **`ort-download-binaries`** (**default**): Automatically downloads prebuilt ONNXRuntime binaries for supported platforms
//! - **`ort-load-dynamic`**: Dynamic linking to ONNXRuntime libraries ([Guide](https://ort.pyke.io/setup/linking#dynamic-linking))
//! - **`video`**: Enable video stream reading and writing (via [video-rs](https://github.com/oddity-ai/video-rs) and [minifb](https://github.com/emoon/rust_minifb))
//! - **`cuda`**: NVIDIA CUDA GPU acceleration support
//! - **`tensorrt`**: NVIDIA TensorRT optimization for inference acceleration
//! - **`coreml`**: Apple CoreML acceleration for macOS/iOS devices
//! - **`openvino`**: Intel OpenVINO toolkit for CPU/GPU/VPU acceleration
//! - **`onednn`**: Intel oneDNN (formerly MKL-DNN) for CPU optimization
//! - **`directml`**: Microsoft DirectML for Windows GPU acceleration
//! - **`xnnpack`**: Google XNNPACK for mobile and edge device optimization
//! - **`rocm`**: AMD ROCm platform for GPU acceleration
//! - **`cann`**: Huawei CANN (Compute Architecture for Neural Networks) support
//! - **`rknpu`**: Rockchip NPU acceleration
//! - **`acl`**: Arm Compute Library for Arm processors
//! - **`nnapi`**: Android Neural Networks API support
//! - **`armnn`**: Arm NN inference engine
//! - **`tvm`**: Apache TVM tensor compiler stack
//! - **`qnn`**: Qualcomm Neural Network SDK
//! - **`migraphx`**: AMD MIGraphX for GPU acceleration
//! - **`vitis`**: Xilinx Vitis AI for FPGA acceleration
//! - **`azure`**: Azure Machine Learning integration
//!

pub mod core;
/// Model Zoo
#[cfg(any(feature = "ort-download-binaries", feature = "ort-load-dynamic"))]
pub mod models;
#[macro_use]
mod results;
pub mod viz;

pub use core::*;
pub use minifb::Key;
#[cfg(any(feature = "ort-download-binaries", feature = "ort-load-dynamic"))]
pub use models::*;
pub use results::*;
pub use viz::*;
