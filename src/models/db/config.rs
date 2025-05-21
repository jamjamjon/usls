/// Model configuration for [DB](https://github.com/MhLiao/DB) and [PaddleOCR-Det](https://github.com/PaddlePaddle/PaddleOCR)
impl crate::Config {
    pub fn db() -> Self {
        Self::default()
            .with_name("db")
            .with_model_ixx(0, 0, (1, 1, 8).into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, (608, 960, 1600).into())
            .with_model_ixx(0, 3, (608, 960, 1600).into())
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_normalize(true)
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_db_binary_thresh(0.2)
            .with_class_confs(&[0.35])
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

    fn ppocr_det_v5() -> Self {
        Self::db()
            .with_model_ixx(0, 2, (608, 640, 1600).into())
            .with_model_ixx(0, 3, (608, 640, 1600).into())
    }

    pub fn ppocr_det_v5_mobile() -> Self {
        Self::ppocr_det_v5().with_model_file("ppocr-v5-mobile.onnx")
    }

    pub fn ppocr_det_v5_server() -> Self {
        Self::ppocr_det_v5().with_model_file("ppocr-v5-server.onnx")
    }

    pub fn db2() -> Self {
        Self::db()
            .with_image_mean(&[0.798, 0.785, 0.772])
            .with_image_std(&[0.264, 0.2749, 0.287])
    }

    pub fn db_mobilenet_v3_large() -> Self {
        Self::db2().with_model_file("felixdittrich92-mobilenet-v3.onnx")
    }

    pub fn db_mobilenet_v3_large_u8() -> Self {
        Self::db2()
            .with_model_file("https://github.com/felixdittrich92/OnnxTR/releases/download/v0.2.0/db_mobilenet_v3_large_static_8_bit-535a6f25.onnx")
    }

    pub fn db_resnet34() -> Self {
        Self::db2().with_model_file("felixdittrich92-r34.onnx")
    }

    pub fn db_resnet34_u8() -> Self {
        Self::db2()
            .with_model_file("https://github.com/felixdittrich92/OnnxTR/releases/download/v0.1.2/db_resnet34_static_8_bit-027e2c7f.onnx")
    }

    pub fn db_resnet50() -> Self {
        Self::db2().with_model_file("felixdittrich92-r50.onnx")
    }

    pub fn db_resnet50_u8() -> Self {
        Self::db2()
            .with_model_file("https://github.com/felixdittrich92/OnnxTR/releases/download/v0.1.2/db_resnet50_static_8_bit-09a6104f.onnx")
    }
}
