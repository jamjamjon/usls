use crate::{
    ResizeMode, NAMES_COCO_80, NAMES_PICODET_LAYOUT_17, NAMES_PICODET_LAYOUT_3,
    NAMES_PICODET_LAYOUT_5,
};

/// Model configuration for `PicoDet`
impl crate::Config {
    pub fn picodet() -> Self {
        Self::default()
            .with_name("picodet")
            .with_batch_size_all(1) // TODO: ONNX model's batch size seems always = 1
            .with_model_ixx(0, 2, 640.into())
            .with_model_ixx(0, 3, 640.into())
            .with_model_ixx(1, 0, (1, 1, 8).into())
            .with_model_ixx(1, 1, 2.into())
            .with_resize_mode(ResizeMode::FitAdaptive)
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_normalize(true)
            .with_class_confs(&[0.5])
    }

    pub fn picodet_l_coco() -> Self {
        Self::picodet()
            .with_model_file("l-coco.onnx")
            .with_class_names(&NAMES_COCO_80)
    }

    pub fn picodet_layout_1x() -> Self {
        Self::picodet()
            .with_model_file("layout-1x.onnx")
            .with_class_names(&NAMES_PICODET_LAYOUT_5)
    }

    pub fn picodet_l_layout_3cls() -> Self {
        Self::picodet()
            .with_model_file("l-layout-3cls.onnx")
            .with_class_names(&NAMES_PICODET_LAYOUT_3)
    }

    pub fn picodet_l_layout_17cls() -> Self {
        Self::picodet()
            .with_model_file("l-layout-17cls.onnx")
            .with_class_names(&NAMES_PICODET_LAYOUT_17)
    }
}
