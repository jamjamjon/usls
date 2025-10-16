use crate::{Config, ResizeMode, Scale, Task, NAMES_YOLOE_4585};

impl Config {
    pub fn yoloe() -> Self {
        Self::default()
            .with_name("yoloe")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 640.into())
            .with_model_ixx(0, 3, 640.into())
            .with_resize_mode(ResizeMode::FitAdaptive)
            .with_resize_filter("CatmullRom")
            .with_task(Task::InstanceSegmentation)
    }

    pub fn yoloe_seg_pf() -> Self {
        Self::yolo()
            .with_class_names(&NAMES_YOLOE_4585)
            .with_task(Task::InstanceSegmentation)
    }

    pub fn yoloe_v8s_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(8.into())
            .with_scale(Scale::S)
            .with_model_file("yoloe-v8s-seg-pf.onnx")
    }

    pub fn yoloe_v8m_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(8.into())
            .with_scale(Scale::M)
            .with_model_file("yoloe-v8m-seg-pf.onnx")
    }

    pub fn yoloe_v8l_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(8.into())
            .with_scale(Scale::L)
            .with_model_file("yoloe-v8l-seg-pf.onnx")
    }

    pub fn yoloe_11s_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(11.into())
            .with_scale(Scale::S)
            .with_model_file("yoloe-11s-seg-pf.onnx")
    }

    pub fn yoloe_11m_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(11.into())
            .with_scale(Scale::M)
            .with_model_file("yoloe-11m-seg-pf.onnx")
    }

    pub fn yoloe_11l_seg_pf() -> Self {
        Self::yoloe_seg_pf()
            .with_version(11.into())
            .with_scale(Scale::L)
            .with_model_file("yoloe-11l-seg-pf.onnx")
    }

    fn yoloe_seg_tp() -> Self {
        Self::yoloe()
            .with_batch_size_all(1)
            .with_nc(80)
            .with_model_ixx(1, 1, (1, 80, 300).into()) // max nc
            .with_textual_encoder_file("mobileclip/blt-textual.onnx")
            .with_model_max_length(77)
            .with_textual_encoder_ixx(0, 1, 77.into())
            .with_tokenizer_file("clip/tokenizer.json")
            .with_tokenizer_config_file("clip/tokenizer_config.json")
            .with_special_tokens_map_file("clip/special_tokens_map.json")
    }

    pub fn yoloe_v8s_seg_tp() -> Self {
        Self::yoloe_seg_tp()
            .with_version(8.into())
            .with_scale(Scale::S)
            .with_model_file("yoloe-v8s-seg-prompt.onnx")
    }

    pub fn yoloe_v8m_seg_tp() -> Self {
        Self::yoloe_seg_tp()
            .with_version(8.into())
            .with_scale(Scale::M)
            .with_model_file("yoloe-v8m-seg-prompt.onnx")
    }

    pub fn yoloe_v8l_seg_tp() -> Self {
        Self::yoloe_seg_tp()
            .with_version(8.into())
            .with_scale(Scale::L)
            .with_model_file("yoloe-v8l-seg-prompt.onnx")
    }

    pub fn yoloe_11s_seg_tp() -> Self {
        Self::yoloe_seg_tp()
            .with_version(11.into())
            .with_scale(Scale::S)
            .with_model_file("yoloe-11s-seg-prompt.onnx")
    }

    pub fn yoloe_11m_seg_tp() -> Self {
        Self::yoloe_seg_tp()
            .with_version(11.into())
            .with_scale(Scale::M)
            .with_model_file("yoloe-11m-seg-prompt.onnx")
    }

    pub fn yoloe_11l_seg_tp() -> Self {
        Self::yoloe_seg_tp()
            .with_version(11.into())
            .with_scale(Scale::L)
            .with_model_file("yoloe-11l-seg-prompt.onnx")
    }

    pub fn yoloe_seg_vp() -> Self {
        Self::yoloe()
            .with_batch_size_all(1)
            .with_nc(80)
            .with_model_ixx(1, 1, (1, 80, 300).into()) // max nc
            .with_visual_encoder_ixx(0, 0, 1.into())
            .with_visual_encoder_ixx(0, 1, 3.into())
            .with_visual_encoder_ixx(0, 2, 640.into()) // h
            .with_visual_encoder_ixx(0, 3, 640.into()) // w
            .with_visual_encoder_ixx(1, 0, 1.into())
            .with_visual_encoder_ixx(1, 1, (1, 80, 300).into())
            .with_visual_encoder_ixx(1, 2, 80.into()) // 1 / 8 * h
            .with_visual_encoder_ixx(1, 3, 80.into()) // 1 / 8 * w
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 640.into())
            .with_model_ixx(0, 3, 640.into())
    }

    pub fn yoloe_v8s_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(8.into())
            .with_scale(Scale::S)
            .with_visual_encoder_file("yoloe-v8s-savpe.onnx")
            .with_model_file("yoloe-v8s-seg-prompt.onnx")
    }

    pub fn yoloe_v8m_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(8.into())
            .with_scale(Scale::M)
            .with_visual_encoder_file("yoloe-v8m-savpe.onnx")
            .with_model_file("yoloe-v8m-seg-prompt.onnx")
    }

    pub fn yoloe_v8l_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(8.into())
            .with_scale(Scale::L)
            .with_visual_encoder_file("yoloe-v8l-savpe.onnx")
            .with_model_file("yoloe-v8l-seg-prompt.onnx")
    }

    pub fn yoloe_11s_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(11.into())
            .with_scale(Scale::S)
            .with_visual_encoder_file("yoloe-11s-savpe.onnx")
            .with_model_file("yoloe-11s-seg-prompt.onnx")
    }

    pub fn yoloe_11m_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(11.into())
            .with_scale(Scale::M)
            .with_visual_encoder_file("yoloe-11m-savpe.onnx")
            .with_model_file("yoloe-11m-seg-prompt.onnx")
    }

    pub fn yoloe_11l_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(11.into())
            .with_scale(Scale::L)
            .with_visual_encoder_file("yoloe-11l-savpe.onnx")
            .with_model_file("yoloe-11l-seg-prompt.onnx")
    }
}
