/// Model configuration for `SVTR`
impl crate::Config {
    pub fn svtr() -> Self {
        Self::default()
            .with_name("svtr")
            .with_model_ixx(0, 0, (1, 1, 8).into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 48.into())
            .with_model_ixx(0, 3, (320, 960, 3200).into())
            .with_resize_mode(crate::ResizeMode::FitHeight)
            .with_padding_value(0)
            .with_normalize(true)
            .with_class_confs(&[0.2])
    }

    pub fn svtr_ch() -> Self {
        Self::svtr().with_vocab_txt("svtr/vocab-v1-ppocr-rec-ch.txt")
    }

    pub fn svtr_en() -> Self {
        Self::svtr().with_vocab_txt("svtr/vocab-v1-ppocr-rec-en.txt")
    }

    pub fn ppocr_rec_v3_ch() -> Self {
        Self::svtr_ch().with_model_file("ppocr-v3-ch.onnx")
    }

    pub fn ppocr_rec_v4_ch() -> Self {
        Self::svtr_ch().with_model_file("ppocr-v4-ch.onnx")
    }

    pub fn ppocr_rec_v3_en() -> Self {
        Self::svtr_en().with_model_file("ppocr-v3-en.onnx")
    }

    pub fn ppocr_rec_v4_en() -> Self {
        Self::svtr_en().with_model_file("ppocr-v4-en.onnx")
    }

    pub fn ppocr_rec_v4_server_ch() -> Self {
        Self::svtr_ch().with_model_file("ppocr-v4-server-ch.onnx")
    }

    pub fn svtr_v2_server_ch() -> Self {
        Self::svtr_ch().with_model_file("v2-server-ch.onnx")
    }

    pub fn repsvtr_ch() -> Self {
        Self::svtr_ch().with_model_file("repsvtr-ch.onnx")
    }

    pub fn svtr_v2_teacher_ch() -> Self {
        Self::svtr_ch().with_model_file("v2-distill-teacher-ch.onnx")
    }

    pub fn svtr_v2_student_ch() -> Self {
        Self::svtr_ch().with_model_file("v2-distill-student-ch.onnx")
    }

    fn ppocr_rec_v5() -> Self {
        Self::svtr().with_vocab_txt("svtr/vocab_v5_ppocr_rec.txt")
    }

    pub fn ppocr_rec_v5_mobile() -> Self {
        Self::ppocr_rec_v5().with_model_file("ppocr-v5-mobile.onnx")
    }

    pub fn ppocr_rec_v5_server() -> Self {
        Self::ppocr_rec_v5().with_model_file("ppocr-v5-server.onnx")
    }
}
