/// Model configuration for `DB`
impl crate::Options {
    pub fn db() -> Self {
        Self::default()
            .with_model_name("db")
            .with_model_ixx(0, 0, (1, 1, 8).into())
            .with_model_ixx(0, 2, (608, 960, 1600).into())
            .with_model_ixx(0, 3, (608, 960, 1600).into())
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_normalize(true)
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_class_confs(&[0.4])
            .with_min_width(5.0)
            .with_min_height(12.0)
    }

    pub fn ppocr_det_v3_ch() -> Self {
        Self::db().with_model_file("ppocr-v3-ch.onnx")
    }

    pub fn ppocr_det_v4_ch() -> Self {
        Self::db().with_model_file("ppocr-v4-ch.onnx")
    }

    pub fn ppocr_det_v4_server_ch() -> Self {
        Self::db().with_model_file("ppocr-v4-server-ch.onnx")
    }
}
