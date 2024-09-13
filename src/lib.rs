//! **`usls`** is a Rust library integrated with **ONNXRuntime** that provides a collection of state-of-the-art models for **Computer Vision** and **Vision-Language** tasks, including:

//! - **YOLO Models**: [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv9](https://github.com/WongKinYiu/yolov9), [YOLOv10](https://github.com/THU-MIG/yolov10)
//! - **SAM Models**: [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/segment-anything-2), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [EdgeSAM](https://github.com/chongzhou96/EdgeSAM), [SAM-HQ](https://github.com/SysCV/sam-hq), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
//! - **Vision Models**: [RTDETR](https://arxiv.org/abs/2304.08069), [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo), [DB](https://arxiv.org/abs/1911.08947), [SVTR](https://arxiv.org/abs/2205.00159), [Depth-Anything-v1-v2](https://github.com/LiheYoung/Depth-Anything), [DINOv2](https://github.com/facebookresearch/dinov2), [MODNet](https://github.com/ZHKKKe/MODNet), [Sapiens](https://arxiv.org/abs/2408.12569)
//! - **Vision-Language Models**: [CLIP](https://github.com/openai/CLIP), [BLIP](https://arxiv.org/abs/2201.12086), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [YOLO-World](https://github.com/AILab-CVC/YOLO-World)
//!
//!
//!
//! # Quick Start
//!
//! The following demo shows how to build model to run and annotate the results.
//!
//! ```rust, no_run
//! use usls::{models::YOLO, Annotator, DataLoader, Options, Vision, YOLOTask, YOLOVersion};
//!
//!
//! fn main() -> anyhow::Result<()> {
//!     // Build model with Options
//!     let options = Options::new()
//!         .with_trt(0)
//!         .with_model("yolo/v8-m-dyn.onnx")?
//!         .with_yolo_version(YOLOVersion::V8)     // YOLOVersion: V5, V6, V7, V8, V9, V10, RTDETR
//!         .with_yolo_task(YOLOTask::Detect)       // YOLOTask: Classify, Detect, Pose, Segment, Obb
//!         .with_i00((1, 1, 4).into())
//!         .with_i02((0, 640, 640).into())
//!         .with_i03((0, 640, 640).into())
//!         .with_confs(&[0.2]);
//!     let mut model = YOLO::new(options)?;
//!
//!     // Build DataLoader to load image(s), video, stream
//!     let dl = DataLoader::new(
//!         "./assets/bus.jpg", // local image
//!         // "images/bus.jpg",  // remote image
//!         // "../set-negs",  // local images (from folder)
//!         // "../hall.mp4",  // local video
//!         // "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",  // remote video
//!         // "rtsp://admin:kkasd1234@192.168.2.217:554/h264/ch1/",  // stream
//!     )?
//!     .with_batch(3)  // iterate with batch_size = 3
//!     .build()?;
//!
//!     // Build annotator
//!     let annotator = Annotator::new().with_saveout("YOLO-DataLoader");
//!
//!     // Run and Annotate images
//!     for (xs, _) in dl {
//!         let ys = model.forward(&xs, false)?;
//!         annotator.annotate(&xs, &ys);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! Refer to [All Demos Here](https://github.com/jamjamjon/usls/tree/main/examples)
//!
//!
//!
//! # How to use Provided Models for Inference
//!
//! #### 1. Build Model
//!
//! <details>
//! <summary>Click to expand</summary>
//!
//! Using provided [`models`] with [`Options`]
//!
//! ```rust, no_run
//! use usls::{ models::YOLO, Annotator, DataLoader, Options, Vision};
//!
//! let options = Options::default()
//!     .with_yolo_version(YOLOVersion::V8)  // YOLOVersion: V5, V6, V7, V8, V9, V10, RTDETR
//!     .with_yolo_task(YOLOTask::Detect)    // YOLOTask: Classify, Detect, Pose, Segment, Obb
//!     .with_model("xxxx.onnx")?;
//! let mut model = YOLO::new(options)?;
//! ```
//!
//! - Choose Execute Provider: `CUDA`(by default), `TensorRT`, or `CoreML`
//!
//! ```rust, no_run
//! let options = Options::default()
//!     .with_cuda(0)
//!     // .with_trt(0)
//!     // .with_coreml(0)
//!     // .with_cpu();
//! ```
//!
//! - Dynamic Input Shapes
//! If your model has dynamic shapes, you need pre-specified it with [`MinOptMax`].
//!
//! `with_ixy()` means the y-th axis of the x-th input. e.g., `i00` is the first axis of the 1st input, batch usually
//!
//! ```rust, no_run
//! let options = Options::default()
//!     .with_i00((1, 2, 4).into()) // batch(min=1, opt=2, max=4)
//!     .with_i02((416, 640, 800).into()) // height(min=416, opt=640, max=800)
//!     .with_i03((416, 640, 800).into()); // width(min=416, opt=640, max=800)
//! ```
//!
//! - Set Confidence Thresholds for Each Category
//!
//! ```rust, no_run
//! let options = Options::default()
//!     .with_confs(&[0.4, 0.15]); // class_0: 0.4, others: 0.15
//! ```
//!
//! - [Optional] Set Class Names
//!
//! ```rust, no_run
//! let options = Options::default()
//!     .with_names(&COCO_CLASS_NAMES_80);
//! ```
//!
//! More options can be found in the [`Options`] documentation.
//!
//! </details>
//!
//!
//! #### 2. Use [`DataLoader`] to load `Image(s)`, `Video` and `Stream`
//!
//! <details>
//! <summary>Click to expand</summary>
//!
//! - Use [`DataLoader::try_read`] to laod single image
//!
//! You can now load image from local file or remote(Github Release Page)
//!
//! ```rust, no_run
//! let x = DataLoader::try_read("./assets/bus.jpg")?; // from local
//! let x = DataLoader::try_read("images/bus.jpg")?; // from remote
//! ```
//!
//! Of course You can directly use [`image::ImageReader`]
//!
//! ```rust, no_run
//! let x = image::ImageReader::open("myimage.png")?.decode()?;
//! ```
//!
//! - Use [`DataLoader] to load image(s), video, stream
//!
//! ```rust, no_run
//!     // Build DataLoader
//!     let dl = DataLoader::new(
//!         "./assets/bus.jpg", // local image
//!         // "images/bus.jpg",  // remote image
//!         // "../set-negs",  // local images (from folder)
//!         // "../hall.mp4",  // local video
//!         // "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",  // remote video
//!         // "rtsp://admin:kkasd1234@192.168.2.217:554/h264/ch1/",  // stream
//!     )?
//!     .with_batch(3)  // iterate with batch_size = 3
//!     .build()?;
//!
//!     // iterate
//!     for (xs, _) in dl {}
//! ```
//!
//! - Use [`DataLoader::is2v`] to convert images into a video
//!
//! ```rust, no_run
//!     let fps = 24;
//!     let image_folder = "runs/YOLO-DataLoader";
//!     let saveout = ["runs", "is2v"];
//!     let fps = 24;
//!     DataLoader::is2v(image_folder, &saveout, 24)?;
//! ```
//!
//!
//! </details>
//!
//!
//!
//! #### 3. Use [`Annotator`] to annotate images  
//!
//!
//! <details>
//! <summary>Click to expand</summary>
//!
//!
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
//! </details>    
//!
//!
//!
//! #### 4. Run and Annotate
//!
//! <details>
//! <summary>Click to expand</summary>
//!
//!
//! ```rust, no_run
//! for (xs, _paths) in dl {
//!     let ys = model.run(&xs)?;
//!     annotator.annotate(&xs, &ys);
//! }
//! ```
//!
//! </details>    
//!
//!
//! #### 5. Get Results
//!
//! <details>
//! <summary>Click to expand</summary>
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
//!
//! </details>    
//!
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
