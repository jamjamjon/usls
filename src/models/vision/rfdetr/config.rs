use crate::{Task, NAMES_COCO_91, NAMES_OBJECT365_366};

///
/// > # RF-DETR: SOTA Real-Time Object Detection Model
/// >
/// > State-of-the-art real-time object detection with transformer architecture.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [roboflow/rf-detr](https://github.com/roboflow/rf-detr)
/// >
/// > # Model Variants
/// >
/// > ## Object Detection Models
/// > - **rfdetr-nano**: Nano model for ultra-fast inference
/// > - **rfdetr-small**: Small model for efficient detection
/// > - **rfdetr-medium**: Medium model for balanced performance
/// > - **rfdetr-base**: Base model for high accuracy
/// > - **rfdetr-base-obj365**: Base model trained on Objects365 dataset
/// > - **rfdetr-large**: Large model for maximum accuracy
/// > - **rfdetr-large-2026**: Large model (2026 version)
/// > - **rfdetr-xlarge**: Extra-large model for enhanced accuracy
/// > - **rfdetr-2xlarge**: Double extra-large model for maximum performance
/// >
/// > ## Instance Segmentation Models
/// > - **rfdetr-seg-preview**: Preview model with segmentation capabilities
/// > - **rfdetr-seg-nano**: Nano segmentation model for ultra-fast inference
/// > - **rfdetr-seg-small**: Small segmentation model for efficient processing
/// > - **rfdetr-seg-medium**: Medium segmentation model for balanced performance
/// > - **rfdetr-seg-large**: Large segmentation model for high accuracy
/// > - **rfdetr-seg-xlarge**: Extra-large segmentation model for enhanced accuracy
/// > - **rfdetr-seg-2xlarge**: Double extra-large segmentation model for maximum performance
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Object Detection**: 91-class COCO object detection
/// > - [X] **Objects365 Detection**: 366-class Objects365 object detection
/// > - [X] **Instance Segmentation**: Object segmentation with full model range
/// > - [X] **Real-time Performance**: Optimized for real-time applications
/// >
/// Model configuration for `RF-DETR`
///
impl crate::Config {
    /// Base configuration for RF-DETR models
    pub fn rfdetr() -> Self {
        Self::default()
            .with_name("rfdetr")
            .with_model_batch_size_min_opt_max(1, 1, 4)
            .with_resize_mode_type(crate::ResizeModeType::Letterbox) // FitAdaptive
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
            .with_class_names(&NAMES_COCO_91)
            .with_task(Task::ObjectDetection)
            .with_class_confs(&[0.3])
    }

    /// RF-DETR nano model
    pub fn rfdetr_nano() -> Self {
        Self::rfdetr().with_model_file("nano.onnx")
    }

    /// RF-DETR small model
    pub fn rfdetr_small() -> Self {
        Self::rfdetr().with_model_file("small.onnx")
    }

    /// RF-DETR medium model
    pub fn rfdetr_medium() -> Self {
        Self::rfdetr().with_model_file("medium.onnx")
    }

    /// RF-DETR base model
    pub fn rfdetr_base() -> Self {
        Self::rfdetr().with_model_file("base.onnx")
    }

    /// RF-DETR base model trained on Objects365
    pub fn rfdetr_base_obj365() -> Self {
        Self::rfdetr()
            .with_class_names(&NAMES_OBJECT365_366)
            .with_model_file("base-obj365.onnx")
    }

    /// RF-DETR large model
    pub fn rfdetr_large() -> Self {
        Self::rfdetr().with_model_file("large.onnx")
    }

    /// RF-DETR large model (2026 version)
    pub fn rfdetr_large_2026() -> Self {
        Self::rfdetr().with_model_file("large-2026.onnx")
    }

    /// RF-DETR xlarge model
    pub fn rfdetr_xlarge() -> Self {
        Self::rfdetr().with_model_file("xlarge.onnx")
    }

    /// RF-DETR 2xlarge model
    pub fn rfdetr_2xlarge() -> Self {
        Self::rfdetr().with_model_file("2xlarge.onnx")
    }

    /// Base configuration for RF-DETR segmentation
    pub fn rfdetr_seg() -> Self {
        Self::rfdetr().with_task(Task::InstanceSegmentation)
    }

    /// RF-DETR seg preview model
    pub fn rfdetr_seg_preview() -> Self {
        Self::rfdetr_seg().with_model_file("seg-preview.onnx")
    }

    /// RF-DETR seg nano model
    pub fn rfdetr_seg_nano() -> Self {
        Self::rfdetr_seg().with_model_file("seg-nano.onnx")
    }

    /// RF-DETR seg small model
    pub fn rfdetr_seg_small() -> Self {
        Self::rfdetr_seg().with_model_file("seg-small.onnx")
    }

    /// RF-DETR seg medium model
    pub fn rfdetr_seg_medium() -> Self {
        Self::rfdetr_seg().with_model_file("seg-medium.onnx")
    }

    /// RF-DETR seg large model
    pub fn rfdetr_seg_large() -> Self {
        Self::rfdetr_seg().with_model_file("seg-large.onnx")
    }

    /// RF-DETR seg xlarge model
    pub fn rfdetr_seg_xlarge() -> Self {
        Self::rfdetr_seg().with_model_file("seg-xlarge.onnx")
    }

    /// RF-DETR seg 2xlarge model
    pub fn rfdetr_seg_2xlarge() -> Self {
        Self::rfdetr_seg().with_model_file("seg-2xlarge.onnx")
    }
}
