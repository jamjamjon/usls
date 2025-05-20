use crate::NAMES_COCO_80;

/// Model configuration for `RT-DETR`
impl crate::Config {
    pub fn rtdetr() -> Self {
        Self::default()
            .with_name("rtdetr")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 640.into())
            .with_model_ixx(0, 3, 640.into())
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_class_confs(&[0.5])
            .with_class_names(&NAMES_COCO_80)
    }

    pub fn rtdetr_v1_r18vd_coco() -> Self {
        Self::rtdetr().with_model_file("v1-r18vd-coco.onnx")
    }

    pub fn rtdetr_v2_s_coco() -> Self {
        Self::rtdetr().with_model_file("v2-s-coco.onnx")
    }

    pub fn rtdetr_v2_ms_coco() -> Self {
        Self::rtdetr().with_model_file("v2-ms-coco.onnx")
    }

    pub fn rtdetr_v2_m_coco() -> Self {
        Self::rtdetr().with_model_file("v2-m-coco.onnx")
    }

    pub fn rtdetr_v2_l_coco() -> Self {
        Self::rtdetr().with_model_file("v2-l-coco.onnx")
    }

    pub fn rtdetr_v2_x_coco() -> Self {
        Self::rtdetr().with_model_file("v2-x-coco.onnx")
    }
}
