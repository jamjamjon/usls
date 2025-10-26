use crate::{Task, NAMES_COCO_91, NAMES_OBJECT365_366};

/// Model configuration for `RF-DETR`
impl crate::Config {
    pub fn rfdetr() -> Self {
        Self::default()
            .with_name("rfdetr")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 560.into())
            .with_model_ixx(0, 3, 560.into())
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_class_names(&NAMES_COCO_91)
            .with_task(Task::ObjectDetection)
            .with_class_confs(&[0.3])
    }

    pub fn rfdetr_nano() -> Self {
        Self::rfdetr().with_model_file("nano.onnx")
    }

    pub fn rfdetr_small() -> Self {
        Self::rfdetr().with_model_file("small.onnx")
    }

    pub fn rfdetr_medium() -> Self {
        Self::rfdetr().with_model_file("medium.onnx")
    }

    pub fn rfdetr_base() -> Self {
        Self::rfdetr().with_model_file("base.onnx")
    }

    pub fn rfdetr_base_obj365() -> Self {
        Self::rfdetr()
            .with_class_names(&NAMES_OBJECT365_366)
            .with_model_file("base-obj365.onnx")
    }

    pub fn rfdetr_large() -> Self {
        Self::rfdetr().with_model_file("large.onnx")
    }

    pub fn rfdetr_seg_preview() -> Self {
        Self::rfdetr()
            .with_task(Task::InstanceSegmentation)
            .with_model_file("seg-preview.onnx")
    }
}
