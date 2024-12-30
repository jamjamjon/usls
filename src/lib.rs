//! **usls** is a Rust library integrated with **ONNXRuntime** that provides a collection of state-of-the-art models for **Computer Vision** and **Vision-Language** tasks, including:
//!
//! - **YOLO Models**: [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv9](https://github.com/WongKinYiu/yolov9), [YOLOv10](https://github.com/THU-MIG/yolov10)
//! - **SAM Models**: [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/segment-anything-2), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [EdgeSAM](https://github.com/chongzhou96/EdgeSAM), [SAM-HQ](https://github.com/SysCV/sam-hq), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
//! - **Vision Models**: [RTDETR](https://arxiv.org/abs/2304.08069), [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo), [DB](https://arxiv.org/abs/1911.08947), [SVTR](https://arxiv.org/abs/2205.00159), [Depth-Anything-v1-v2](https://github.com/LiheYoung/Depth-Anything), [DINOv2](https://github.com/facebookresearch/dinov2), [MODNet](https://github.com/ZHKKKe/MODNet), [Sapiens](https://arxiv.org/abs/2408.12569)
//! - **Vision-Language Models**: [CLIP](https://github.com/openai/CLIP), [BLIP](https://arxiv.org/abs/2201.12086), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [Florence2](https://arxiv.org/abs/2311.06242)
//!

mod misc;
pub mod models;
mod xy;

pub use misc::*;
pub use models::*;
pub use xy::*;
