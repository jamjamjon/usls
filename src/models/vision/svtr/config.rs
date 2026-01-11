///
/// > # SVTR: Scene Text Recognition with a Single Visual Model
/// >
/// > Single visual model for scene text recognition with transformer-based architecture.
/// >
/// > # Paper & Code
/// >
/// > - **Official**: [PaddleOCR SVTRv2](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/text_recognition/algorithm_rec_svtrv2.html)
/// >
/// > # Model Variants
/// >
/// > - **ppocr-rec-v3-ch**: PaddleOCR v3 Chinese text recognition
/// > - **ppocr-rec-v3-en**: PaddleOCR v3 English text recognition
/// > - **ppocr-rec-v4-ch**: PaddleOCR v4 Chinese text recognition
/// > - **ppocr-rec-v4-en**: PaddleOCR v4 English text recognition
/// > - **ppocr-rec-v4-server-ch**: PaddleOCR v4 server Chinese recognition
/// > - **svtr-v2-server-ch**: SVTR v2 server Chinese recognition
/// > - **repsvtr-ch**: Representative SVTR Chinese recognition
/// > - **svtr-v2-teacher-ch**: SVTR v2 teacher model for distillation
/// > - **svtr-v2-student-ch**: SVTR v2 student model for distillation
/// > - **ppocr-rec-v5-mobile**: PaddleOCR v5 mobile recognition
/// > - **ppocr-rec-v5-server**: PaddleOCR v5 server recognition
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Scene Text Recognition**: End-to-end text recognition
/// > - [X] **Chinese Text Support**: Optimized for Chinese character recognition
/// > - [X] **English Text Support**: English character recognition
/// > - [X] **Knowledge Distillation**: Teacher-student model variants
/// > - [X] **Mobile Optimization**: Lightweight models for mobile devices
/// >
/// Model configuration for `SVTR`
///
impl crate::Config {
    /// Base configuration for SVTR models
    pub fn svtr() -> Self {
        Self::default()
            .with_name("svtr")
            .with_model_ixx(0, 0, (1, 1, 8))
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 48)
            .with_model_ixx(0, 3, (320, 960, 3200))
            .with_resize_mode_type(crate::ResizeModeType::FitHeight)
            .with_padding_value(0)
            .with_normalize(true)
            .with_class_confs(&[0.2])
    }

    fn svtr_ch() -> Self {
        Self::svtr()
            .with_class_names_owned(Self::load_txt_into_vec("svtr/vocab-v1-ppocr-rec-ch.txt"))
    }

    fn svtr_en() -> Self {
        Self::svtr()
            .with_class_names_owned(Self::load_txt_into_vec("svtr/vocab-v1-ppocr-rec-en.txt"))
    }

    /// PaddleOCR v3 Chinese text recognition
    pub fn ppocr_rec_v3_ch() -> Self {
        Self::svtr_ch().with_model_file("ppocr-v3-ch.onnx")
    }

    /// PaddleOCR v4 Chinese text recognition
    pub fn ppocr_rec_v4_ch() -> Self {
        Self::svtr_ch().with_model_file("ppocr-v4-ch.onnx")
    }

    /// PaddleOCR v3 English text recognition
    pub fn ppocr_rec_v3_en() -> Self {
        Self::svtr_en().with_model_file("ppocr-v3-en.onnx")
    }

    /// PaddleOCR v4 English text recognition
    pub fn ppocr_rec_v4_en() -> Self {
        Self::svtr_en().with_model_file("ppocr-v4-en.onnx")
    }

    /// PaddleOCR v4 server Chinese recognition
    pub fn ppocr_rec_v4_server_ch() -> Self {
        Self::svtr_ch().with_model_file("ppocr-v4-server-ch.onnx")
    }

    /// SVTR v2 server Chinese recognition
    pub fn svtr_v2_server_ch() -> Self {
        Self::svtr_ch().with_model_file("v2-server-ch.onnx")
    }

    /// Representative SVTR Chinese recognition
    pub fn repsvtr_ch() -> Self {
        Self::svtr_ch().with_model_file("repsvtr-ch.onnx")
    }

    /// SVTR v2 teacher model for distillation
    pub fn svtr_v2_teacher_ch() -> Self {
        Self::svtr_ch().with_model_file("v2-distill-teacher-ch.onnx")
    }

    /// SVTR v2 student model for distillation
    pub fn svtr_v2_student_ch() -> Self {
        Self::svtr_ch().with_model_file("v2-distill-student-ch.onnx")
    }

    fn ppocr_rec_v5() -> Self {
        Self::svtr().with_class_names_owned(Self::load_txt_into_vec("svtr/vocab_v5_ppocr_rec.txt"))
    }

    /// PaddleOCR v5 mobile recognition
    pub fn ppocr_rec_v5_mobile() -> Self {
        Self::ppocr_rec_v5().with_model_file("ppocr-v5-mobile.onnx")
    }

    /// PaddleOCR v5 server recognition
    pub fn ppocr_rec_v5_server() -> Self {
        Self::ppocr_rec_v5().with_model_file("ppocr-v5-server.onnx")
    }
}
