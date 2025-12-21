//! # usls
//!
//! **usls** is a cross-platform Rust library powered by ONNX Runtime for efficient inference of SOTA vision and vision-language models (typically under 1B parameters).
//!
//! ## 📚 Documentation
//! - [API Documentation](https://docs.rs/usls/latest/usls/)
//! - [Examples](https://github.com/jamjamjon/usls/tree/main/examples)
//!
//! ## ⚡ Cargo Features
//!
//! > ❕ Features in ***italics*** are enabled by default.
//!
//! - ### Runtime & Utilities
//!   - ***`ort-download-binaries`***: Auto-download ONNX Runtime binaries from [pyke](https://ort.pyke.io/perf/execution-providers).
//!   - **`ort-load-dynamic`**: Linking ONNX Runtime by your self. Use this if `pyke` doesn't provide prebuilt binaries for your platform or you want to link your local ONNX Runtime library. See [Linking Guide](https://ort.pyke.io/setup/linking#static-linking) for more details.
//!   - **`viewer`**: Image/video visualization ([minifb](https://github.com/emoon/rust_minifb)). Similar to OpenCV `imshow()`. See [example](./examples/imshow.rs).
//!   - **`video`**: Video I/O support ([video-rs](https://github.com/oddity-ai/video-rs)). Enable this to read/write video streams. See [example](./examples/read_video.rs)
//!   - **`hf-hub`**: Hugging Face Hub support for downloading models from Hugging Face repositories.
//!   - **`tokenizers`**: Tokenizer support for vision-language models. Automatically enabled when using vision-language model features (blip, clip, florence2, grounding-dino, fastvlm, moondream2, owl, smolvlm, trocr, yoloe).
//!
//! - ### Execution Providers
//!   Hardware acceleration for inference.
//!
//!   - **`cuda`**, **`tensorrt`**: NVIDIA GPU acceleration
//!   - **`coreml`**: Apple Silicon acceleration
//!   - **`openvino`**: Intel CPU/GPU/VPU acceleration
//!   - **`onednn`**, **`directml`**, **`xnnpack`**, **`rocm`**, **`cann`**, **`rknpu`**, **`acl`**, **`nnapi`**, **`armnn`**, **`tvm`**, **`qnn`**, **`migraphx`**, **`vitis`**, **`azure`**: Various hardware/platform support
//!
//!   See [ONNX Runtime docs](https://onnxruntime.ai/docs/execution-providers/) and [ORT performance guide](https://ort.pyke.io/perf/execution-providers) for details.
//!
//! - ### Model Selection
//!   Almost each model is a separate feature. Enable only what you need to reduce compile time and binary size.
//!
//!   - *`yolo`*, `sam`, `clip`, `image-classifier`, `dino`, `rtmpose`, `rtdetr`, `db`, ...
//!   - **All models**: `all-models` (enables all model features)
//!
//!   See [Supported Models](https://github.com/jamjamjon/usls#-supported-models) for the complete list with feature names.
//!

#[macro_use]
pub mod processor;
#[macro_use]
pub mod backend;
#[macro_use]
pub mod config;
pub mod dataloader;
#[cfg(feature = "ort")]
pub mod models;
pub mod utils;
#[macro_use]
mod results;
mod tensor;
pub mod viz;

// Backend re-exports (ORT-dependent)
#[cfg(feature = "ort")]
#[doc(inline)]
pub use backend::Engine;
#[cfg(feature = "ort")]
#[doc(inline)]
pub use backend::Xs;
#[cfg(feature = "ort")]
#[doc(inline)]
pub use backend::{EpConfig, MinOptMax, ORTConfig};
#[cfg(feature = "ort")]
#[doc(inline)]
pub use backend::{OrtInput, OrtInputs};

// utils re-exports
#[doc(hidden)]
pub use utils::*;
#[doc(inline)]
pub use utils::{DType, Device, Dir, DynConf, Scale, Task, Version};

pub use tensor::*;

#[doc(hidden)]
pub use config::*;

// Dataloader re-exports
#[doc(hidden)]
pub use dataloader::*;
#[doc(inline)]
pub use dataloader::{DataLoader, Hub, Image, Location, MediaType};

// Processor re-exports
#[doc(inline)]
pub use dataloader::ResizeMode;
#[cfg(feature = "viewer")]
pub use minifb::Key;
// Model and names re-exports
#[cfg(feature = "ort")]
#[doc(hidden)]
pub use models::*;

pub use backend::*;
#[cfg(feature = "cuda")]
#[doc(inline)]
pub use processor::{compute_convolution_1d, CudaImageProcessContext, CudaPreprocessor};
#[doc(inline)]
pub use processor::{
    ImagePlan, ImageProcessor, ImageProcessorConfig, ImageTensorLayout, ResizeAlg, ResizeFilter,
};
#[cfg(feature = "vlm")]
#[doc(inline)]
pub use processor::{LogitsSampler, TextProcessor, TextProcessorConfig};
pub use results::*;
#[doc(inline)]
pub use results::{
    SKELETON_COCO_19, SKELETON_COCO_65, SKELETON_COLOR_COCO_19, SKELETON_COLOR_COCO_65,
    SKELETON_COLOR_HALPE_27, SKELETON_COLOR_HAND_21, SKELETON_HALPE_27, SKELETON_HAND_21,
};
#[doc(hidden)]
pub use viz::*;
