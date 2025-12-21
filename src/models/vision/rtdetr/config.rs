use crate::NAMES_COCO_80;

/// Model configuration for `RT-DETR`
impl crate::Config {
    pub fn rtdetr() -> Self {
        Self::default()
            .with_name("rtdetr")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 640)
            .with_model_ixx(0, 3, 640)
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_class_confs(&[0.35])
            .with_class_names(&NAMES_COCO_80)
    }

    pub fn rtdetr_v1_r18() -> Self {
        Self::rtdetr()
            .with_version(1.into())
            .with_model_file("v1-r18.onnx")
    }

    pub fn rtdetr_v1_r18_obj365() -> Self {
        Self::rtdetr()
            .with_version(1.into())
            .with_model_file("v1-r18-obj365.onnx")
    }

    pub fn rtdetr_v1_r34() -> Self {
        Self::rtdetr()
            .with_version(1.into())
            .with_model_file("v1-r34.onnx")
    }

    pub fn rtdetr_v1_r50() -> Self {
        Self::rtdetr()
            .with_version(1.into())
            .with_model_file("v1-r50.onnx")
    }

    pub fn rtdetr_v1_r50_obj365() -> Self {
        Self::rtdetr()
            .with_version(1.into())
            .with_model_file("v1-r50-obj365.onnx")
    }

    pub fn rtdetr_v1_r101() -> Self {
        Self::rtdetr()
            .with_version(1.into())
            .with_model_file("v1-r101.onnx")
    }

    pub fn rtdetr_v1_r101_obj365() -> Self {
        Self::rtdetr()
            .with_version(1.into())
            .with_model_file("v1-r101-obj365.onnx")
    }

    pub fn rtdetr_v2_s() -> Self {
        Self::rtdetr()
            .with_version(2.into())
            .with_model_file("v2-s.onnx")
    }

    pub fn rtdetr_v2_ms() -> Self {
        Self::rtdetr()
            .with_version(2.into())
            .with_model_file("v2-ms.onnx")
    }

    pub fn rtdetr_v2_m() -> Self {
        Self::rtdetr()
            .with_version(2.into())
            .with_model_file("v2-m.onnx")
    }

    pub fn rtdetr_v2_l() -> Self {
        Self::rtdetr()
            .with_version(2.into())
            .with_model_file("v2-l.onnx")
    }

    pub fn rtdetr_v2_x() -> Self {
        Self::rtdetr()
            .with_version(2.into())
            .with_model_file("v2-x.onnx")
    }

    pub fn rtdetr_v4_s() -> Self {
        Self::rtdetr()
            .with_version(4.into())
            .with_model_file("v4-s.onnx")
    }

    pub fn rtdetr_v4_m() -> Self {
        Self::rtdetr()
            .with_version(4.into())
            .with_model_file("v4-m.onnx")
    }

    pub fn rtdetr_v4_l() -> Self {
        Self::rtdetr()
            .with_version(4.into())
            .with_model_file("v4-l.onnx")
    }

    pub fn rtdetr_v4_x() -> Self {
        Self::rtdetr()
            .with_version(4.into())
            .with_model_file("v4-x.onnx")
    }
}
