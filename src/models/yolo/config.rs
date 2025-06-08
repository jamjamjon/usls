use crate::{
    models::YOLOPredsFormat, Config, ResizeMode, Scale, Task, NAMES_COCO_80,
    NAMES_COCO_KEYPOINTS_17, NAMES_DOTA_V1_15, NAMES_IMAGENET_1K, NAMES_YOLOE_4585,
    NAMES_YOLO_DOCLAYOUT_10,
};

impl Config {
    /// Creates a base YOLO configuration with common settings.
    ///
    /// Sets up default input dimensions (640x640) and image processing parameters.
    pub fn yolo() -> Self {
        Self::default()
            .with_name("yolo")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 640.into())
            .with_model_ixx(0, 3, 640.into())
            .with_resize_mode(ResizeMode::FitAdaptive)
            .with_resize_filter("CatmullRom")
    }

    /// Creates a configuration for YOLO image classification.
    ///
    /// Configures the model for ImageNet classification with:
    /// - 224x224 input size
    /// - Exact resize mode with bilinear interpolation
    /// - ImageNet 1000 class names
    pub fn yolo_classify() -> Self {
        Self::yolo()
            .with_task(Task::ImageClassification)
            .with_model_ixx(0, 2, 224.into())
            .with_model_ixx(0, 3, 224.into())
            .with_resize_mode(ResizeMode::FitExact)
            .with_resize_filter("Bilinear")
            .with_class_names(&NAMES_IMAGENET_1K)
    }

    /// Creates a configuration for YOLO object detection.
    ///
    /// Configures the model for COCO dataset object detection with 80 classes.
    pub fn yolo_detect() -> Self {
        Self::yolo()
            .with_task(Task::ObjectDetection)
            .with_class_names(&NAMES_COCO_80)
    }

    /// Creates a configuration for YOLO pose estimation.
    ///
    /// Configures the model for human keypoint detection with 17 COCO keypoints.
    pub fn yolo_pose() -> Self {
        Self::yolo()
            .with_task(Task::KeypointsDetection)
            .with_keypoint_names(&NAMES_COCO_KEYPOINTS_17)
    }

    /// Creates a configuration for YOLO instance segmentation.
    ///
    /// Configures the model for COCO dataset instance segmentation with 80 classes.
    pub fn yolo_segment() -> Self {
        Self::yolo()
            .with_task(Task::InstanceSegmentation)
            .with_class_names(&NAMES_COCO_80)
    }

    /// Creates a configuration for YOLO oriented object detection.
    ///
    /// Configures the model for detecting rotated objects with:
    /// - 1024x1024 input size
    /// - DOTA v1 dataset classes
    pub fn yolo_obb() -> Self {
        Self::yolo()
            .with_model_ixx(0, 2, 1024.into())
            .with_model_ixx(0, 3, 1024.into())
            .with_task(Task::OrientedObjectDetection)
            .with_class_names(&NAMES_DOTA_V1_15)
    }

    /// Creates a configuration for document layout analysis using YOLOv10.
    ///
    /// Configures the model for detecting document structure elements with:
    /// - Variable input size up to 1024x1024
    /// - 10 document layout classes
    pub fn doclayout_yolo_docstructbench() -> Self {
        Self::yolo_detect()
            .with_version(10.into())
            .with_model_ixx(0, 2, (640, 1024, 1024).into())
            .with_model_ixx(0, 3, (640, 1024, 1024).into())
            .with_class_confs(&[0.4])
            .with_class_names(&NAMES_YOLO_DOCLAYOUT_10)
            .with_model_file("doclayout-docstructbench.onnx") // TODO: batch_size > 1
    }

    /// Creates a base YOLOE configuration with 4585 classes.
    ///
    /// Configures the model for instance segmentation with a large class vocabulary.
    pub fn yoloe() -> Self {
        Self::yolo()
            .with_task(Task::InstanceSegmentation)
            .with_class_names(&NAMES_YOLOE_4585)
    }
    /// Creates a configuration for YOLOE-v8s segmentation model.
    /// Uses the small variant of YOLOv8 architecture.
    pub fn yoloe_v8s_seg_pf() -> Self {
        Self::yoloe()
            .with_version(8.into())
            .with_scale(Scale::S)
            .with_model_file("yoloe-v8s-seg-pf.onnx")
    }

    /// Creates a configuration for YOLOE-v8m segmentation model.
    /// Uses the medium variant of YOLOv8 architecture.
    pub fn yoloe_v8m_seg_pf() -> Self {
        Self::yoloe()
            .with_version(8.into())
            .with_scale(Scale::M)
            .with_model_file("yoloe-v8m-seg-pf.onnx")
    }

    /// Creates a configuration for YOLOE-v8l segmentation model.
    /// Uses the large variant of YOLOv8 architecture.
    pub fn yoloe_v8l_seg_pf() -> Self {
        Self::yoloe()
            .with_version(8.into())
            .with_scale(Scale::L)
            .with_model_file("yoloe-v8l-seg-pf.onnx")
    }

    /// Creates a configuration for YOLOE-11s segmentation model.
    /// Uses the small variant of YOLOv11 architecture.
    pub fn yoloe_11s_seg_pf() -> Self {
        Self::yoloe()
            .with_version(11.into())
            .with_scale(Scale::S)
            .with_model_file("yoloe-11s-seg-pf.onnx")
    }

    pub fn yoloe_11m_seg_pf() -> Self {
        Self::yoloe()
            .with_version(11.into())
            .with_scale(Scale::M)
            .with_model_file("yoloe-v8m-seg-pf.onnx")
    }

    pub fn yoloe_11l_seg_pf() -> Self {
        Self::yoloe()
            .with_version(11.into())
            .with_scale(Scale::L)
            .with_model_file("yoloe-11l-seg-pf.onnx")
    }

    pub fn fastsam_s() -> Self {
        Self::yolo_segment()
            .with_class_names(&["object"])
            .with_scale(Scale::S)
            .with_version(8.into())
            .with_model_file("FastSAM-s.onnx")
    }

    pub fn fastsam_x() -> Self {
        Self::yolo_segment()
            .with_class_names(&["object"])
            .with_scale(Scale::X)
            .with_version(8.into())
            .with_model_file("FastSAM-x.onnx")
    }

    pub fn ultralytics_rtdetr_l() -> Self {
        Self::yolo_detect()
            .with_yolo_preds_format(YOLOPredsFormat::n_a_cxcywh_clss_n())
            .with_scale(Scale::L)
            .with_model_file("rtdetr-l.onnx")
    }

    pub fn ultralytics_rtdetr_x() -> Self {
        Self::yolo_detect()
            .with_yolo_preds_format(YOLOPredsFormat::n_a_cxcywh_clss_n())
            .with_scale(Scale::X)
            .with_model_file("rtdetr-x.onnx")
    }
}
