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
/// > - **rfdetr-nano**: Nano model for ultra-fast inference
/// > - **rfdetr-small**: Small model for efficient detection
/// > - **rfdetr-medium**: Medium model for balanced performance
/// > - **rfdetr-base**: Base model for high accuracy
/// > - **rfdetr-base-obj365**: Base model trained on Objects365 dataset
/// > - **rfdetr-large**: Large model for maximum accuracy
/// > - **rfdetr-seg-preview**: Preview model with segmentation capabilities
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Object Detection**: 91-class COCO object detection
/// > - [X] **Objects365 Detection**: 366-class Objects365 object detection
/// > - [X] **Instance Segmentation**: Object segmentation (preview)
/// > - [X] **Real-time Performance**: Optimized for real-time applications
/// >
/// Model configuration for `RF-DETR`
///
impl crate::Config {
    /// Base configuration for RF-DETR models
    pub fn rfdetr() -> Self {
        Self::default()
            .with_name("rfdetr")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 560)
            .with_model_ixx(0, 3, 560)
            .with_resize_mode_type(crate::ResizeModeType::FitAdaptive)
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
            .with_class_names(&NAMES_COCO_91)
            .with_task(Task::ObjectDetection)
            .with_class_confs(&[0.3])
    }

    /// Nano model for ultra-fast inference
    pub fn rfdetr_nano() -> Self {
        Self::rfdetr().with_model_file("nano.onnx")
    }

    /// Small model for efficient detection
    pub fn rfdetr_small() -> Self {
        Self::rfdetr().with_model_file("small.onnx")
    }

    /// Medium model for balanced performance
    pub fn rfdetr_medium() -> Self {
        Self::rfdetr().with_model_file("medium.onnx")
    }

    /// Base model for high accuracy
    pub fn rfdetr_base() -> Self {
        Self::rfdetr().with_model_file("base.onnx")
    }

    /// Base model trained on Objects365 dataset
    pub fn rfdetr_base_obj365() -> Self {
        Self::rfdetr()
            .with_class_names(&NAMES_OBJECT365_366)
            .with_model_file("base-obj365.onnx")
    }

    /// Large model for maximum accuracy
    pub fn rfdetr_large() -> Self {
        Self::rfdetr().with_model_file("large.onnx")
    }

    /// Preview model with segmentation capabilities
    pub fn rfdetr_seg_preview() -> Self {
        Self::rfdetr()
            .with_task(Task::InstanceSegmentation)
            .with_model_file("seg-preview.onnx")
    }
}
