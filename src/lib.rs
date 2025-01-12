//! **usls** is a Rust library integrated with **ONNXRuntime**, offering a suite of advanced models for **Computer Vision** and **Vision-Language** tasks, including:
//!
//! - **YOLO Models**: [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv9](https://github.com/WongKinYiu/yolov9), [YOLOv10](https://github.com/THU-MIG/yolov10), [YOLO11](https://github.com/ultralytics/ultralytics)
//! - **SAM Models**: [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/segment-anything-2), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [EdgeSAM](https://github.com/chongzhou96/EdgeSAM), [SAM-HQ](https://github.com/SysCV/sam-hq), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
//! - **Vision Models**: [RT-DETR](https://arxiv.org/abs/2304.08069), [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo), [Depth-Anything](https://github.com/LiheYoung/Depth-Anything), [DINOv2](https://github.com/facebookresearch/dinov2), [MODNet](https://github.com/ZHKKKe/MODNet), [Sapiens](https://arxiv.org/abs/2408.12569), [DepthPro](https://github.com/apple/ml-depth-pro), [FastViT](https://github.com/apple/ml-fastvit), [BEiT](https://github.com/microsoft/unilm/tree/master/beit), [MobileOne](https://github.com/apple/ml-mobileone)
//! - **Vision-Language Models**: [CLIP](https://github.com/openai/CLIP), [jina-clip-v1](https://huggingface.co/jinaai/jina-clip-v1), [BLIP](https://arxiv.org/abs/2201.12086), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [Florence2](https://arxiv.org/abs/2311.06242)
//! - **OCR Models**: [DB](https://arxiv.org/abs/1911.08947), [FAST](https://github.com/czczup/FAST), [SVTR](https://arxiv.org/abs/2205.00159), [SLANet](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/table_recognition/algorithm_table_slanet.html), [TrOCR](https://huggingface.co/microsoft/trocr-base-printed), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)  
//! - **And more...**
//!
//! ## ⛳️ Cargo Features
//!
//! By default, **none of the following features are enabled**. You can enable them as needed:
//!
//! - **`auto`**: Automatically downloads prebuilt ONNXRuntime binaries from Pyke’s CDN for supported platforms.
//!
//!   - If disabled, you'll need to [compile `ONNXRuntime` from source](https://github.com/microsoft/onnxruntime) or [download a precompiled package](https://github.com/microsoft/onnxruntime/releases), and then [link it manually](https://ort.pyke.io/setup/linking).
//!
//!     <details>
//!     <summary>👉 For Linux or macOS Users</summary>
//!
//!     - Download from the [Releases page](https://github.com/microsoft/onnxruntime/releases).
//!     - Set up the library path by exporting the `ORT_DYLIB_PATH` environment variable:
//!       ```shell
//!       export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so.1.20.1
//!       ```
//!
//!     </details>
//! - **`ffmpeg`**: Adds support for video streams, real-time frame visualization, and video export.
//!
//!   - Powered by [video-rs](https://github.com/oddity-ai/video-rs) and [minifb](https://github.com/emoon/rust_minifb). For any issues related to `ffmpeg` features, please refer to the issues of these two crates.
//! - **`cuda`**: Enables the NVIDIA TensorRT provider.
//! - **`trt`**: Enables the NVIDIA TensorRT provider.
//! - **`mps`**: Enables the Apple CoreML provider.
//!
//! ## 🎈 Example
//!
//! ```Shell
//! cargo run -r -F cuda --example svtr -- --device cuda
//! ```
//!
//! All examples are located in the [examples](https://github.com/jamjamjon/usls/tree/main/examples) directory.

mod misc;
pub mod models;
mod xy;

pub use misc::*;
pub use models::*;
pub use xy::*;
