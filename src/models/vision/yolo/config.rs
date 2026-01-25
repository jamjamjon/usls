use crate::{
    models::YOLOPredsFormat, Config, Scale, Task, NAMES_COCO_80, NAMES_COCO_KEYPOINTS_17,
    NAMES_DOTA_V1_0_15, NAMES_IMAGENET_1K, NAMES_YOLO_DOCLAYOUT_10,
};

///
/// > # YOLO: You Only Look Once
/// >
/// > Real-time object detection system with comprehensive computer vision capabilities.
/// >
/// > # Paper & Code
/// >
/// > - **YOLOv5**: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
/// > - **YOLOv6**: [meituan/YOLOv6](https://github.com/meituan/YOLOv6)
/// > - **YOLOv7**: [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
/// > - **YOLOv8/v11**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
/// > - **YOLOv9**: [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
/// > - **YOLOv10**: [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
/// > - **YOLOv12**:  [sunsmarterjie/YOLOv12](https://github.com/sunsmarterjie/yolov12)
/// > - **YOLOv13**: [iMoonLab/YOLOv13](https://github.com/iMoonLab/yolov13)
/// >
/// > # Model Variants
/// >
/// > - **yolo-classify**: Image classification with ImageNet 1000 classes
/// > - **yolo-detect**: Object detection with COCO 80 classes
/// > - **yolo-pose**: Pose estimation with 17 COCO keypoints
/// > - **yolo-segment**: Instance segmentation with COCO 80 classes
/// > - **yolo-obb**: Oriented object detection with DOTA 15 classes
/// > - **doclayout-yolo-docstructbench**: Document layout analysis with 10 classes
/// > - **ultralytics-rtdetr-l**: Large RT-DETR model
/// > - **ultralytics-rtdetr-x**: Extra large RT-DETR model
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Object Detection**: Real-time multi-class object detection
/// > - [X] **Image Classification**: 1000-class ImageNet classification
/// > - [X] **Pose Estimation**: 17-keypoint human pose detection
/// > - [X] **Instance Segmentation**: Pixel-level object segmentation
/// > - [X] **Oriented Detection**: Rotated bounding box detection
/// > - [X] **Document Layout**: Document structure analysis
/// >
/// Model configuration for `YOLO`
///
impl Config {
    /// Base configuration for YOLO models with common settings
    pub fn yolo() -> Self {
        Self::default()
            .with_name("yolo")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 640)
            .with_model_ixx(0, 3, 640)
            .with_resize_alg(crate::ResizeAlg::Interpolation(
                crate::ResizeFilter::Bilinear,
            ))
            // .with_resize_mode_type(crate::ResizeModeType::FitAdaptive)
            .with_resize_mode_type(crate::ResizeModeType::Letterbox)
    }

    /// Image classification with ImageNet 1000 classes
    pub fn yolo_classify() -> Self {
        Self::yolo()
            .with_task(Task::ImageClassification)
            .with_model_ixx(0, 2, 224)
            .with_model_ixx(0, 3, 224)
            .with_resize_mode_type(crate::ResizeModeType::FitExact)
            .with_class_names(&NAMES_IMAGENET_1K)
    }

    /// Object detection with COCO 80 classes
    pub fn yolo_detect() -> Self {
        Self::yolo()
            .with_task(Task::ObjectDetection)
            .with_class_names(&NAMES_COCO_80)
    }

    /// Pose estimation with 17 COCO keypoints
    pub fn yolo_pose() -> Self {
        Self::yolo()
            .with_task(Task::KeypointsDetection)
            .with_keypoint_names(&NAMES_COCO_KEYPOINTS_17)
    }

    /// Instance segmentation with COCO 80 classes
    pub fn yolo_segment() -> Self {
        Self::yolo()
            .with_task(Task::InstanceSegmentation)
            .with_class_names(&NAMES_COCO_80)
    }

    /// Oriented object detection with DOTA 15 classes
    pub fn yolo_obb() -> Self {
        Self::yolo()
            .with_model_ixx(0, 2, 1024)
            .with_model_ixx(0, 3, 1024)
            .with_task(Task::OrientedObjectDetection)
            .with_class_names(&NAMES_DOTA_V1_0_15)
    }

    /// Document layout analysis using YOLOv10
    pub fn doclayout_yolo_docstructbench() -> Self {
        Self::yolo_detect()
            .with_version(10.into())
            .with_model_ixx(0, 2, (640, 1024, 1024))
            .with_model_ixx(0, 3, (640, 1024, 1024))
            .with_class_confs(&[0.4])
            .with_class_names(&NAMES_YOLO_DOCLAYOUT_10)
            .with_model_file("doclayout-docstructbench.onnx")
    }

    /// Large RT-DETR model
    pub fn ultralytics_rtdetr_l() -> Self {
        Self::yolo_detect()
            .with_yolo_preds_format(YOLOPredsFormat::n_a_cxcywh_clss_n())
            .with_scale(Scale::L)
            .with_model_file("rtdetr-l.onnx")
    }

    /// Extra large RT-DETR model
    pub fn ultralytics_rtdetr_x() -> Self {
        Self::yolo_detect()
            .with_yolo_preds_format(YOLOPredsFormat::n_a_cxcywh_clss_n())
            .with_scale(Scale::X)
            .with_model_file("rtdetr-x.onnx")
    }
}
