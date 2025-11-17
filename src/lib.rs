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
//!   - **`slsl`**: SLSL tensor library support. Automatically enabled when using `yolo` or `clip` features.
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

pub mod core;
#[cfg(any(feature = "ort-download-binaries", feature = "ort-load-dynamic"))]
pub mod models;
#[macro_use]
mod results;
#[cfg(feature = "mot")]
pub mod mot;
pub mod viz;

#[cfg(any(feature = "ort-download-binaries", feature = "ort-load-dynamic"))]
#[doc(inline)]
pub use core::Engine;
#[cfg(all(
    any(feature = "ort-download-binaries", feature = "ort-load-dynamic"),
    feature = "slsl"
))]
#[doc(inline)]
pub use core::OrtEngine;
#[doc(hidden)]
pub use core::*;
#[doc(inline)]
pub use core::{
    Config, DType, DataLoader, Device, Dir, DynConf, HardwareConfig, Hub, Image, ImageTensorLayout,
    Location, LogitsSampler, MediaType, ORTConfig, Processor, ProcessorConfig, ResizeMode, Scale,
    Task, Version, Xs, X,
};
#[cfg(feature = "viewer")]
pub use minifb::Key;
#[cfg(any(feature = "ort-download-binaries", feature = "ort-load-dynamic"))]
#[doc(inline)]
pub use models::names::{
    NAMES_COCO_80, NAMES_COCO_91, NAMES_COCO_KEYPOINTS_133, NAMES_COCO_KEYPOINTS_17,
    NAMES_HAND_KEYPOINTS_21, NAMES_IMAGENET_1K, NAMES_OBJECT365, NAMES_OBJECT365_366,
};
#[cfg(all(
    any(feature = "ort-download-binaries", feature = "ort-load-dynamic"),
    feature = "sapiens"
))]
#[doc(inline)]
pub use models::NAMES_BODY_PARTS_28;
#[cfg(all(
    any(feature = "ort-download-binaries", feature = "ort-load-dynamic"),
    feature = "rtmpose"
))]
#[doc(inline)]
pub use models::NAMES_HALPE_KEYPOINTS_26;
#[cfg(all(
    any(feature = "ort-download-binaries", feature = "ort-load-dynamic"),
    feature = "yoloe"
))]
#[doc(inline)]
pub use models::NAMES_YOLOE_4585;
#[cfg(any(feature = "ort-download-binaries", feature = "ort-load-dynamic"))]
#[doc(hidden)]
pub use models::*;
#[cfg(all(
    any(feature = "ort-download-binaries", feature = "ort-load-dynamic"),
    feature = "yolo"
))]
#[doc(inline)]
pub use models::{NAMES_DOTA_V1_15, NAMES_DOTA_V1_5_16, NAMES_YOLO_DOCLAYOUT_10};
#[cfg(all(
    any(feature = "ort-download-binaries", feature = "ort-load-dynamic"),
    feature = "picodet"
))]
#[doc(inline)]
pub use models::{NAMES_PICODET_LAYOUT_17, NAMES_PICODET_LAYOUT_3, NAMES_PICODET_LAYOUT_5};
#[cfg(feature = "mot")]
#[doc(hidden)]
pub use mot::*;
pub use results::*;
#[doc(inline)]
pub use results::{
    SKELETON_COCO_19, SKELETON_COCO_65, SKELETON_COLOR_COCO_19, SKELETON_COLOR_COCO_65,
    SKELETON_COLOR_HALPE_27, SKELETON_COLOR_HAND_21, SKELETON_HALPE_27, SKELETON_HAND_21,
};
#[doc(hidden)]
pub use viz::*;
