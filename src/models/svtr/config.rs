/// Model configuration for `SVTR`
impl crate::Options {
    pub fn svtr() -> Self {
        Self::default()
            .with_model_name("svtr")
            .with_model_ixx(0, 0, (1, 1, 8).into())
            .with_model_ixx(0, 2, 48.into())
            .with_model_ixx(0, 3, (320, 960, 1600).into())
            .with_resize_mode(crate::ResizeMode::FitHeight)
            .with_padding_value(0)
            .with_normalize(true)
            .with_class_confs(&[0.2])
            .with_vocab_txt("vocab-v1-ppocr-rec-ch.txt")
    }

    pub fn ppocr_rec_v3_ch() -> Self {
        Self::svtr().with_model_file("ppocr-v3-ch.onnx")
    }

    pub fn ppocr_rec_v4_ch() -> Self {
        Self::svtr().with_model_file("ppocr-v4-ch.onnx")
    }

    pub fn ppocr_rec_v4_server_ch() -> Self {
        Self::svtr().with_model_file("ppocr-v4-server-ch.onnx")
    }

    pub fn svtr_v2_server_ch() -> Self {
        Self::svtr().with_model_file("v2-server-ch.onnx")
    }

    pub fn repsvtr_ch() -> Self {
        Self::svtr().with_model_file("repsvtr-ch.onnx")
    }

    pub fn svtr_v2_teacher_ch() -> Self {
        Self::svtr().with_model_file("v2-distill-teacher-ch.onnx")
    }

    pub fn svtr_v2_student_ch() -> Self {
        Self::svtr().with_model_file("v2-distill-student-ch.onnx")
    }
}
