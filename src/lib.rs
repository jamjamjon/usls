//! A Rust library integrated with ONNXRuntime, providing a collection of Computer Vision and Vision-Language models.
//!
//! # Supported Models
//!
//! - [YOLOv5](https://github.com/ultralytics/yolov5): Object Detection, Instance Segmentation, Classification
//! - [YOLOv6](https://github.com/meituan/YOLOv6): Object Detection
//! - [YOLOv7](https://github.com/WongKinYiu/yolov7): Object Detection
//! - [YOLOv8](https://github.com/ultralytics/ultralytics): Object Detection, Instance Segmentation, Classification, Oriented Object Detection, Keypoint Detection
//! - [YOLOv9](https://github.com/WongKinYiu/yolov9): Object Detection
//! - [YOLOv10](https://github.com/THU-MIG/yolov10): Object Detection
//! - [RT-DETR](https://arxiv.org/abs/2304.08069): Object Detection
//! - [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM): Instance Segmentation
//! - [YOLO-World](https://github.com/AILab-CVC/YOLO-World): Object Detection
//! - [DINOv2](https://github.com/facebookresearch/dinov2): Vision-Self-Supervised
//! - [CLIP](https://github.com/openai/CLIP): Vision-Language
//! - [BLIP](https://github.com/salesforce/BLIP): Vision-Language
//! - [DB](https://arxiv.org/abs/1911.08947): Text Detection
//! - [SVTR](https://arxiv.org/abs/2205.00159): Text Recognition
//! - [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo): Keypoint Detection
//! - [YOLOPv2](https://arxiv.org/abs/2208.11434): Panoptic Driving Perception
//! - [Depth-Anything (v1, v2)](https://github.com/LiheYoung/Depth-Anything): Monocular Depth Estimation
//! - [MODNet](https://github.com/ZHKKKe/MODNet): Image Matting
//!
//! # Examples
//!
//! [All Demos Here](https://github.com/jamjamjon/usls/tree/main/examples)
//!
//! # Using Provided Models for Inference
//!
//! #### 1. Build Model
//! Using provided [`models`] with [`Options`]
//!
//! ```rust, no_run
//! use usls::{coco, models::YOLO, Annotator, DataLoader, Options, Vision};
//!
//! let options = Options::default()
//!     .with_yolo_version(YOLOVersion::V8)  // YOLOVersion: V5, V6, V7, V8, V9, V10, RTDETR
//!     .with_yolo_task(YOLOTask::Detect)    // YOLOTask: Classify, Detect, Pose, Segment, Obb
//!     .with_model("xxxx.onnx")?;
//! let mut model = YOLO::new(options)?;
//! ```
//!
//! - Use `CUDA`, `TensorRT`, or `CoreML`
//!
//! ```rust, no_run
//! let options = Options::default()
//!     .with_cuda(0) // using CUDA by default
//!     // .with_trt(0)
//!     // .with_coreml(0)
//!     // .with_cpu();
//! ```
//!
//! - Dynamic Input Shapes
//!
//! ```rust, no_run
//! let options = Options::default()
//!     .with_i00((1, 2, 4).into()) // dynamic batch
//!     .with_i02((416, 640, 800).into()) // dynamic height
//!     .with_i03((416, 640, 800).into()); // dynamic width
//! ```
//!
//! - Set Confidence Thresholds for Each Category
//!
//! ```rust, no_run
//! let options = Options::default()
//!     .with_confs(&[0.4, 0.15]); // class_0: 0.4, others: 0.15
//! ```
//!
//! - Set Class Names
//!
//! ```rust, no_run
//! let options = Options::default()
//!     .with_names(&coco::NAMES_80);
//! ```
//!
//! More options can be found in the [`Options`] documentation.
//!
//! #### 2. Load Images
//!
//! Ensure that the input image is RGB type.
//!
//! - Using [`image::ImageReader`] or [`DataLoader`] to Load One Image
//!
//! ```rust, no_run
//! let x = vec![DataLoader::try_read("./assets/bus.jpg")?];
//! // or
//! let x = image::ImageReader::open("myimage.png")?.decode()?;
//! ```
//!
//! - Using [`DataLoader`] to Load a Batch of Images
//!
//! ```rust, no_run
//! let dl = DataLoader::default()
//!     .with_batch(4)
//!     .load("./assets")?;
//! ```
//!
//! #### 3. (Optional) Annotate Results with [`Annotator`]
//!
//! ```rust, no_run
//! let annotator = Annotator::default();
//! ```
//!
//! - Set Saveout Name
//!
//! ```rust, no_run
//! let annotator = Annotator::default()
//!     .with_saveout("YOLOs");
//! ```
//!     
//! - Set Bboxes Line Width
//!
//! ```rust, no_run
//! let annotator = Annotator::default()
//!     .with_bboxes_thickness(4);
//! ```
//!  
//! - Disable Mask Plotting
//!  
//! ```rust, no_run
//! let annotator = Annotator::default()
//!     .without_masks(true);
//! ```
//!   
//! More options can be found in the [`Annotator`] documentation.
//!
//!     
//! #### 4. Run and Annotate
//!
//! ```rust, no_run
//! for (xs, _paths) in dl {
//!     let ys = model.run(&xs)?;
//!     annotator.annotate(&xs, &ys);
//! }
//! ```
//!
//! #### 5. Get Results
//!
//! The inference outputs of provided models will be saved to a [`Vec<Y>`].
//!
//! - For Example, Get Detection Bboxes with `y.bboxes()`
//!
//! ```rust, no_run
//! let ys = model.run(&xs)?;
//! for y in ys {
//!     // bboxes
//!     if let Some(bboxes) = y.bboxes() {
//!         for bbox in bboxes {
//!             println!(
//!                 "Bbox: {}, {}, {}, {}, {}, {}",
//!                 bbox.xmin(),
//!                 bbox.ymin(),
//!                 bbox.xmax(),
//!                 bbox.ymax(),
//!                 bbox.confidence(),
//!                 bbox.id(),
//!             );
//!         }
//!     }
//! }
//! ```
//!
//! # Also, You Can Implement Your Own Model with [`OrtEngine`] and [`Options`]
//!
//! [`OrtEngine`] provides ONNX model loading, metadata parsing, dry_run, inference, and other functions, supporting EPs such as CUDA, TensorRT, CoreML, etc. You can use it as the ONNXRuntime engine for building models.
//!
//! Refer to [Demo: Depth-Anything](https://github.com/jamjamjon/usls/blob/main/src/models/depth_anything.rs) for more details.

mod core;
pub mod models;
mod utils;
mod ys;

pub use core::*;
pub use models::*;
pub use utils::*;
pub use ys::*;
