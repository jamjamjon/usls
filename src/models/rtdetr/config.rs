use crate::models::COCO_CLASS_NAMES_80;

/// Model configuration for `RT-DETR`
impl crate::Options {
    pub fn rtdetr() -> Self {
        Self::default()
            .with_model_name("rtdetr")
            .with_batch_size(1)
            .with_model_ixx(0, 2, 640.into())
            .with_model_ixx(0, 3, 640.into())
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_normalize(true)
            .with_class_confs(&[0.5])
            .with_class_names(&COCO_CLASS_NAMES_80)
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

    pub fn dfine() -> Self {
        Self::rtdetr().with_model_name("dfine")
    }

    pub fn dfine_n_coco() -> Self {
        Self::dfine().with_model_file("n-coco.onnx")
    }

    pub fn dfine_s_coco() -> Self {
        Self::dfine().with_model_file("s-coco.onnx")
    }

    pub fn dfine_m_coco() -> Self {
        Self::dfine().with_model_file("m-coco.onnx")
    }

    pub fn dfine_l_coco() -> Self {
        Self::dfine().with_model_file("l-coco.onnx")
    }

    pub fn dfine_x_coco() -> Self {
        Self::dfine().with_model_file("x-coco.onnx")
    }

    pub fn dfine_s_coco_obj365() -> Self {
        Self::dfine().with_model_file("s-obj2coco.onnx")
    }

    pub fn dfine_m_coco_obj365() -> Self {
        Self::dfine().with_model_file("m-obj2coco.onnx")
    }

    pub fn dfine_l_coco_obj365() -> Self {
        Self::dfine().with_model_file("l-obj2coco.onnx")
    }

    pub fn dfine_x_coco_obj365() -> Self {
        Self::dfine().with_model_file("x-obj2coco.onnx")
    }
}
