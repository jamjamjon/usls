/// Model configuration for `SVTR`
impl crate::Options {
    pub fn svtr() -> Self {
        Self::default()
            .with_model_name("svtr")
            .with_model_ixx(0, 0, (1, 1, 8).into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 48.into())
            .with_model_ixx(0, 3, (320, 960, 1600).into())
            .with_resize_mode(crate::ResizeMode::FitHeight)
            .with_padding_value(0)
            .with_normalize(true)
            .with_class_confs(&[0.2])
    }

    pub fn svtr_ch() -> Self {
        Self::svtr().with_vocab_txt("vocab-v1-ppocr-rec-ch.txt")
    }

    pub fn svtr_en() -> Self {
        Self::svtr().with_vocab_txt("vocab-v1-ppocr-rec-en.txt")
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
}

#[derive(aksr::Builder, Debug, Clone)]
pub struct SVTRConfig {
    pub model: crate::ModelConfig,
    pub processor: crate::ProcessorConfig,
    pub class_confs: Vec<f32>,
}

impl Default for SVTRConfig {
    fn default() -> Self {
        Self {
            model: crate::ModelConfig::default()
                .with_name("svtr")
                .with_ixx(0, 0, (1, 1, 8).into())
                .with_ixx(0, 1, 3.into())
                .with_ixx(0, 2, 48.into())
                .with_ixx(0, 3, (320, 960, 1600).into()),
            processor: crate::ProcessorConfig::default()
                .with_resize_mode(crate::ResizeMode::FitHeight)
                .with_padding_value(0)
                .with_resize_filter("Bilinear"),
            class_confs: vec![0.2f32],
        }
    }
}

use crate::{impl_model_config_methods, impl_process_config_methods};

impl_model_config_methods!(SVTRConfig, model);
impl_process_config_methods!(SVTRConfig, processor);

impl SVTRConfig {
    pub fn svtr_ch() -> Self {
        Self::default().with_vocab_txt("vocab-v1-ppocr-rec-ch.txt")
    }

    pub fn svtr_en() -> Self {
        Self::default().with_vocab_txt("vocab-v1-ppocr-rec-en.txt")
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
}
