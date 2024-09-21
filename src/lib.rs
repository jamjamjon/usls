//! **usls** is a Rust library integrated with **ONNXRuntime** that provides a collection of state-of-the-art models for **Computer Vision** and **Vision-Language** tasks, including:
//!
//! - **YOLO Models**: [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv9](https://github.com/WongKinYiu/yolov9), [YOLOv10](https://github.com/THU-MIG/yolov10)
//! - **SAM Models**: [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/segment-anything-2), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [EdgeSAM](https://github.com/chongzhou96/EdgeSAM), [SAM-HQ](https://github.com/SysCV/sam-hq), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
//! - **Vision Models**: [RTDETR](https://arxiv.org/abs/2304.08069), [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo), [DB](https://arxiv.org/abs/1911.08947), [SVTR](https://arxiv.org/abs/2205.00159), [Depth-Anything-v1-v2](https://github.com/LiheYoung/Depth-Anything), [DINOv2](https://github.com/facebookresearch/dinov2), [MODNet](https://github.com/ZHKKKe/MODNet), [Sapiens](https://arxiv.org/abs/2408.12569)
//! - **Vision-Language Models**: [CLIP](https://github.com/openai/CLIP), [BLIP](https://arxiv.org/abs/2201.12086), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [Florence2](https://arxiv.org/abs/2311.06242)
//!
//! # Examples
//!
//! Refer to [All Runnable Demos](https://github.com/jamjamjon/usls/tree/main/examples)
//!
//! # Quick Start
//!
//! The following demo shows how to build a `YOLO` with [`Options`], load `image(s)`, `video` and `stream` with [`DataLoader`], and annotate the model's inference results with [`Annotator`].
//!
//! ```ignore
//! use usls::{models::YOLO, Annotator, DataLoader, Options, Vision, YOLOTask, YOLOVersion};
//!
//! fn main() -> anyhow::Result<()> {
//!     // Build model with Options
//!     let options = Options::new()
//!         .with_trt(0)
//!         .with_model("yolo/v8-m-dyn.onnx")?
//!         .with_yolo_version(YOLOVersion::V8) // YOLOVersion: V5, V6, V7, V8, V9, V10, RTDETR
//!         .with_yolo_task(YOLOTask::Detect) // YOLOTask: Classify, Detect, Pose, Segment, Obb
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
//!     let annotator = Annotator::new().with_saveout("YOLO-Demo");
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

//! # What's More
//!
//! This guide covers the process of using provided models for inference, including how to build a model, load data, annotate results, and retrieve the outputs. Click the sections below to expand for detailed instructions.
//!
//! <details>
//! <summary>Build the Model</summary>
//!
//! To build a model, you can use the provided [models] with [Options]:
//!
//! ```ignore
//! use usls::{models::YOLO, Annotator, DataLoader, Options, Vision};
//!
//! let options = Options::default()
//!     .with_yolo_version(YOLOVersion::V8)  // YOLOVersion: V5, V6, V7, V8, V9, V10, RTDETR
//!     .with_yolo_task(YOLOTask::Detect)    // YOLOTask: Classify, Detect, Pose, Segment, Obb
//!     .with_model("xxxx.onnx")?;
//! let mut model = YOLO::new(options)?;
//! ```
//!
//! **And there're many options provided by [Options]**
//!
//! - **Choose Execution Provider:**  
//!   Select `CUDA` (default), `TensorRT`, or `CoreML`:
//!
//! ```ignore
//! let options = Options::default()
//!     .with_cuda(0)
//!     // .with_trt(0)
//!     // .with_coreml(0)
//!     // .with_cpu();
//! ```
//!
//! - **Dynamic Input Shapes:**  
//!   Specify dynamic shapes with [MinOptMax]:
//!
//! ```ignore
//! let options = Options::default()
//!     .with_i00((1, 2, 4).into()) // batch(min=1, opt=2, max=4)
//!     .with_i02((416, 640, 800).into()) // height(min=416, opt=640, max=800)
//!     .with_i03((416, 640, 800).into()); // width(min=416, opt=640, max=800)
//! ```
//!
//! - **Set Confidence Thresholds:**  
//!   Adjust thresholds for each category:
//!
//! ```ignore
//! let options = Options::default()
//!     .with_confs(&[0.4, 0.15]); // class_0: 0.4, others: 0.15
//! ```
//!
//! - **Set Class Names:**  
//!   Provide class names if needed:
//!
//! ```ignore
//! let options = Options::default()
//!     .with_names(&COCO_CLASS_NAMES_80);
//! ```
//!
//! **More options are detailed in the [Options] documentation.**  
//!   
//!
//! </details>
//!
//! <details>
//! <summary>Load Images, Video and Stream</summary>
//!
//! - **Load a Single Image**  
//! Use [DataLoader::try_read] to load an image from a local file or remote source:
//!
//! ```ignore
//! let x = DataLoader::try_read("./assets/bus.jpg")?; // from local
//! let x = DataLoader::try_read("images/bus.jpg")?; // from remote
//! ```
//!
//! Alternatively, use [image::ImageReader] directly:
//!
//! ```ignore
//! let x = image::ImageReader::open("myimage.png")?.decode()?;
//! ```
//!
//! - **Load Multiple Images, Videos, or Streams**  
//! Create a [DataLoader] instance for batch processing:
//!
//! ```ignore
//! let dl = DataLoader::new(
//!     "./assets/bus.jpg", // local image
//!     // "images/bus.jpg",  // remote image
//!     // "../set-negs",  // local images (from folder)
//!     // "../hall.mp4",  // local video
//!     // "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",  // remote video
//!     // "rtsp://admin:kkasd1234@192.168.2.217:554/h264/ch1/",  // stream
//! )?
//! .with_batch(3)  // iterate with batch_size = 3
//! .build()?;
//!
//! // Iterate through the data
//! for (xs, _) in dl {}
//! ```
//!
//! - **Convert Images to Video**  
//! Use [DataLoader::is2v] to create a video from a sequence of images:
//!
//! ```ignore
//! let fps = 24;
//! let image_folder = "runs/YOLO-DataLoader";
//! let saveout = ["runs", "is2v"];
//! DataLoader::is2v(image_folder, &saveout, fps)?;
//! ```
//!
//! </details>
//!
//! <details>
//! <summary>Annotate Inference Results</summary>
//!
//! - **Create an Annotator Instance**
//!
//! ```ignore
//! let annotator = Annotator::default();
//! ```
//!
//! - **Set Saveout Name:**
//!
//! ```ignore
//! let annotator = Annotator::default()
//!     .with_saveout("YOLOs");
//! ```
//!
//! - **Set Bounding Box Line Width:**
//!
//! ```ignore
//! let annotator = Annotator::default()
//!     .with_bboxes_thickness(4);
//! ```
//!
//! - **Disable Mask Plotting**
//!
//! ```ignore
//! let annotator = Annotator::default()
//!     .without_masks(true);
//! ```
//!
//! - **Perform Inference and nnotate the results**  
//!
//! ```ignore
//! for (xs, _paths) in dl {
//!     let ys = model.run(&xs)?;
//!     annotator.annotate(&xs, &ys);
//! }
//! ```
//!
//! More options are detailed in the [Annotator] documentation.
//!
//! </details>
//!
//! <details>
//! <summary>Retrieve Model's Inference Results</summary>
//!
//! Retrieve the inference outputs, which are saved in a [`Vec<Y>`]:
//!
//! - **Get Detection Bounding Boxes**
//!
//! ```ignore
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
//! </details>
//!
//! <details>
//! <summary>Custom Model Implementation</summary>
//!
//! You can also implement your own model using [OrtEngine] and [Options]. [OrtEngine] supports ONNX model loading, metadata parsing, dry_run, inference, and other functions, with execution providers such as CUDA, TensorRT, CoreML, etc.
//!
//! For more details, refer to the [Demo: Depth-Anything](https://github.com/jamjamjon/usls/blob/main/src/models/depth_anything.rs).
//!
//! </details>

mod core;
pub mod models;
mod utils;
mod ys;

pub use core::*;
pub use models::*;
pub use utils::*;
pub use ys::*;
