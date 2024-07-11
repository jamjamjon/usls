//! A Rust library integrated with ONNXRuntime, providing a collection of Computer Vison and Vision-Language models.
//!
//! [`OrtEngine`] provides ONNX model loading, metadata parsing, dry_run, inference and other functions, supporting EPs such as CUDA, TensorRT, CoreML, etc. You can use it as the ONNXRuntime engine for building models.
//!
//!
//!
//!

//! # Supported models
//! |                               Model                               |         Task / Type         |    
//! | :---------------------------------------------------------------: | :-------------------------: |
//! |    [YOLOv5](https://github.com/ultralytics/yolov5)    |      Object Detection<br />Instance Segmentation<br />Classification      |
//! |    [YOLOv6](https://github.com/meituan/YOLOv6)    |      Object Detection      |
//! |    [YOLOv7](https://github.com/WongKinYiu/yolov7)    |      Object Detection      |  
//! |         [YOLOv8](https://github.com/ultralytics/ultralytics)         | Object Detection<br />Instance Segmentation<br />Classification<br />Oriented Object Detection<br />Keypoint Detection |
//! |            [YOLOv9](https://github.com/WongKinYiu/yolov9)            |      Object Detection      |  
//! |            [YOLOv10](https://github.com/THU-MIG/yolov10)            |      Object Detection      |  
//! |             [RT-DETR](https://arxiv.org/abs/2304.08069)             |      Object Detection      |
//! |         [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)         |    Instance Segmentation    |  
//! |        [YOLO-World](https://github.com/AILab-CVC/YOLO-World)        |      Object Detection      |
//! |         [DINOv2](https://github.com/facebookresearch/dinov2)         |   Vision-Self-Supervised   |  
//! |                [CLIP](https://github.com/openai/CLIP)                |       Vision-Language       |  
//! |              [BLIP](https://github.com/salesforce/BLIP)              |       Vision-Language       |  
//! |                [DB](https://arxiv.org/abs/1911.08947)                |       Text Detection       |
//! |               [SVTR](https://arxiv.org/abs/2205.00159)               |      Text Recognition      |
//! | [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) |     Keypoint Detection     |  
//! |             [YOLOPv2](https://arxiv.org/abs/2208.11434)             | Panoptic Driving Perception |
//! |     [Depth-Anything<br />(v1, v2)](https://github.com/LiheYoung/Depth-Anything)     |    Monocular Depth Estimation    |  
//! |     [MODNet](https://github.com/ZHKKKe/MODNet)     |    Image Matting    |

//! # Examples
//! [All Examples Here](https://github.com/jamjamjon/usls/tree/main/examples)

//! # Use provided models for inference

//! #### 1. Using provided [`models`] with [`Option`]

//! ```Rust, no_run
//! use usls::{coco, models::YOLO, Annotator, DataLoader, Options, Vision};
//!
//! let options = Options::default()
//!     .with_yolo_version(YOLOVersion::V8)  // YOLOVersion: V5, V6, V7, V8, V9, V10, RTDETR
//!     .with_yolo_task(YOLOTask::Detect)  // YOLOTask: Classify, Detect, Pose, Segment, Obb
//!     .with_model("xxxx.onnx")?;
//!     .with_trt(0)
//!     .with_fp16(true)
//!     .with_i00((1, 1, 4).into())
//!     .with_i02((224, 640, 800).into())
//!     .with_i03((224, 640, 800).into())
//!     .with_confs(&[0.4, 0.15]) // class_0: 0.4, others: 0.15
//!     .with_profile(false);
//! let mut model = YOLO::new(options)?;
//! ```

//! #### 2. Load images using [`DataLoader`] or [`image::io::Reader`]
//!
//! ```Rust, no_run
//! // Load one image
//! let x = vec![DataLoader::try_read("./assets/bus.jpg")?];
//!
//! // Load images with batch_size = 4
//! let dl = DataLoader::default()
//!     .with_batch(4)
//!     .load("./assets")?;
//! // Load one image with `image::io::Reader`
//! let x = image::io::Reader::open("myimage.png")?.decode()?
//! ```
//!
//! #### 3. Build annotator using [`Annotator`]
//!
//! ```Rust, no_run
//! let annotator = Annotator::default()
//!     .with_bboxes_thickness(4)
//!     .with_saveout("YOLOs");
//! ```
//!
//! #### 4. Run and annotate
//!
//! ```Rust, no_run
//! for (xs, _paths) in dl {
//!     let ys = model.run(&xs)?;
//!     annotator.annotate(&xs, &ys);
//! }
//! ```
//!
//! #### 5. Parse inference results from [`Vec<Y>`]
//! For example, uou can get detection bboxes with `y.bboxes()`:
//! ```Rust, no_run
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
//!             )
//!         }
//!     }
//! }
//!   ```
//!
//!
//! # Build your own model with [`OrtEngine`]
//!
//! Refer to [Demo: Depth-Anything](https://github.com/jamjamjon/usls/blob/main/src/models/depth_anything.rs)
//!
//!

mod core;
pub mod models;
mod utils;
mod ys;

pub use core::*;
pub use models::*;
pub use utils::*;
pub use ys::*;
