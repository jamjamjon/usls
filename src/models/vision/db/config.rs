///
/// > # DB: Real-time Scene Text Detection
/// >
/// > Real-time arbitrary-shape scene text detector with differentiable binarization.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [MhLiao/DB](https://github.com/MhLiao/DB)
/// > - **PaddleOCR**: [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
/// > - **Paper**: [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/2002.12244)
/// >
/// > # Model Variants
/// >
/// > - **ppocr-v3-ch**: PaddleOCR v3 Chinese text detection
/// > - **ppocr-v4-ch**: PaddleOCR v4 Chinese text detection
/// > - **ppocr-v4-server-ch**: PaddleOCR v4 server Chinese detection
/// > - **ppocr-v5-mobile**: PaddleOCR v5 mobile detection
/// > - **ppocr-v5-server**: PaddleOCR v5 server detection
/// > - **db-mobilenet-v3-large**: MobileNetV3 backbone DB model
/// > - **db-resnet34**: ResNet34 backbone DB model
/// > - **db-resnet50**: ResNet50 backbone DB model
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Scene Text Detection**: Arbitrary-shape text detection
/// > - [X] **Chinese Text Detection**: Chinese character recognition
/// >
/// Model configuration for `DB`
///
impl crate::Config {
    /// Base configuration for DB models
    pub fn db() -> Self {
        Self::default()
            .with_name("db")
            .with_model_ixx(0, 0, (1, 1, 8))
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, (608, 960, 1600))
            .with_model_ixx(0, 3, (608, 960, 1600))
            .with_resize_mode_type(crate::ResizeModeType::FitAdaptive)
            .with_normalize(true)
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
            .with_db_binary_thresh(0.2)
            .with_class_confs(&[0.35])
            .with_min_width(5)
            .with_min_height(12)
    }

    /// PaddleOCR v3 Chinese text detection
    pub fn ppocr_det_v3_ch() -> Self {
        Self::db().with_model_file("ppocr-v3-ch.onnx")
    }

    /// PaddleOCR v4 Chinese text detection
    pub fn ppocr_det_v4_ch() -> Self {
        Self::db().with_model_file("ppocr-v4-ch.onnx")
    }

    /// PaddleOCR v4 server Chinese detection
    pub fn ppocr_det_v4_server_ch() -> Self {
        Self::db().with_model_file("ppocr-v4-server-ch.onnx")
    }

    fn ppocr_det_v5() -> Self {
        Self::db()
            .with_model_ixx(0, 2, (608, 960, 1600))
            .with_model_ixx(0, 3, (608, 960, 1600))
    }

    /// PaddleOCR v5 mobile detection
    pub fn ppocr_det_v5_mobile() -> Self {
        Self::ppocr_det_v5().with_model_file("ppocr-v5-mobile.onnx")
    }

    /// PaddleOCR v5 server detection
    pub fn ppocr_det_v5_server() -> Self {
        Self::ppocr_det_v5().with_model_file("ppocr-v5-server.onnx")
    }

    fn db2() -> Self {
        Self::db()
            .with_image_mean([0.798, 0.785, 0.772])
            .with_image_std([0.264, 0.2749, 0.287])
    }

    /// MobileNetV3 backbone DB model
    pub fn db_mobilenet_v3_large() -> Self {
        Self::db2().with_model_file("felixdittrich92-mobilenet-v3.onnx")
    }

    /// MobileNetV3 backbone DB model (8-bit quantized)
    pub fn db_mobilenet_v3_large_u8() -> Self {
        Self::db2()
            .with_model_file("https://github.com/felixdittrich92/OnnxTR/releases/download/v0.2.0/db_mobilenet_v3_large_static_8_bit-535a6f25.onnx")
    }

    /// ResNet34 backbone DB model
    pub fn db_resnet34() -> Self {
        Self::db2().with_model_file("felixdittrich92-r34.onnx")
    }

    /// ResNet34 backbone DB model (8-bit quantized)
    pub fn db_resnet34_u8() -> Self {
        Self::db2()
            .with_model_file("https://github.com/felixdittrich92/OnnxTR/releases/download/v0.1.2/db_resnet34_static_8_bit-027e2c7f.onnx")
    }

    /// ResNet50 backbone DB model
    pub fn db_resnet50() -> Self {
        Self::db2().with_model_file("felixdittrich92-r50.onnx")
    }

    /// ResNet50 backbone DB model (8-bit quantized)
    pub fn db_resnet50_u8() -> Self {
        Self::db2()
            .with_model_file("https://github.com/felixdittrich92/OnnxTR/releases/download/v0.1.2/db_resnet50_static_8_bit-09a6104f.onnx")
    }
}
