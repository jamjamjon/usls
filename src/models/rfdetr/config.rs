use crate::NAMES_COCO_91;

/// Model configuration for `RT-DETR`
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
    }

    pub fn rfdetr_base() -> Self {
        Self::rfdetr().with_model_file("base.onnx")
    }

    pub fn rfdetr_large() -> Self {
        Self::rfdetr().with_model_file("large.onnx")
    }
}
