use crate::{Config, Scale, Version};

///
/// > # YOLOE: Real-Time Seeing Anything
/// >
/// > Real-time vision-language model for comprehensive object detection and segmentation with text prompts.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe)
/// > - **GitHub**: [ultralytics/YOLOE-26](https://github.com/ultralytics/ultralytics)
/// >
/// > # Model Variants
/// >
/// > - **yoloe-v8s/m/l-seg-tp**: YOLOE v8 text-prompt segmentation models
/// > - **yoloe-11s/m/l-seg-tp**: YOLOE 11 text-prompt segmentation models
/// > - **yoloe-v8s/m/l-seg-vp**: YOLOE v8 visual-prompt segmentation models
/// > - **yoloe-11s/m/l-seg-vp**: YOLOE 11 visual-prompt segmentation models
/// > - **yoloe-26n/m/s/m/l/x-seg-tp**: YOLOE 26 text-prompt segmentation models
/// > - **yoloe-26n/m/s/m/l/x-seg-vp**: YOLOE 26 visual-prompt segmentation models
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Text-Prompt Segmentation**: Segment objects described by text
/// > - [X] **Visual-Prompt Segmentation**: Segment with visual prompts
/// >
/// Model configuration for `YOLOE` with prompt-based inference.
///
impl Config {
    /// Base configuration for YOLOE text-prompt segmentation
    fn yoloe_seg_tp() -> Self {
        Self::yoloe()
            .with_batch_size_all(1)
            .with_nc(80)
            .with_model_ixx(1, 1, (1, 80, 300)) // max nc/ max dets
            .with_model_max_length(77)
            .with_textual_encoder_ixx(0, 1, 77)
            .with_textual_encoder_file("mobileclip/blt-textual.onnx")
            .with_tokenizer_file("clip/tokenizer.json")
            .with_tokenizer_config_file("clip/tokenizer_config.json")
            .with_special_tokens_map_file("clip/special_tokens_map.json")
    }

    /// YOLOE v8 small text-prompt segmentation model
    pub fn yoloe_v8s_seg_tp() -> Self {
        Self::yoloe_seg_tp()
            .with_version(Version::from(8))
            .with_scale(Scale::S)
            .with_model_file("yoloe-v8s-seg-prompt.onnx")
    }

    /// YOLOE v8 medium text-prompt segmentation model
    pub fn yoloe_v8m_seg_tp() -> Self {
        Self::yoloe_seg_tp()
            .with_version(Version::from(8))
            .with_scale(Scale::M)
            .with_model_file("yoloe-v8m-seg-prompt.onnx")
    }

    /// YOLOE v8 large text-prompt segmentation model
    pub fn yoloe_v8l_seg_tp() -> Self {
        Self::yoloe_seg_tp()
            .with_version(Version::from(8))
            .with_scale(Scale::L)
            .with_model_file("yoloe-v8l-seg-prompt.onnx")
    }

    /// YOLOE 11 small text-prompt segmentation model
    pub fn yoloe_11s_seg_tp() -> Self {
        Self::yoloe_seg_tp()
            .with_version(Version::from(11))
            .with_scale(Scale::S)
            .with_model_file("yoloe-11s-seg-prompt.onnx")
    }

    /// YOLOE 11 medium text-prompt segmentation model
    pub fn yoloe_11m_seg_tp() -> Self {
        Self::yoloe_seg_tp()
            .with_version(Version::from(11))
            .with_scale(Scale::M)
            .with_model_file("yoloe-11m-seg-prompt.onnx")
    }

    /// YOLOE 11 large text-prompt segmentation model
    pub fn yoloe_11l_seg_tp() -> Self {
        Self::yoloe_seg_tp()
            .with_version(Version::from(11))
            .with_scale(Scale::L)
            .with_model_file("yoloe-11l-seg-prompt.onnx")
    }

    /// Base configuration for YOLOE 26 text-prompt segmentation
    fn yoloe_26_seg_tp() -> Self {
        Self::yoloe_seg_tp().with_textual_encoder_file("mobileclip2/b-textual.onnx")
    }

    /// YOLOE 26 small text-prompt segmentation model
    pub fn yoloe_26n_seg_tp() -> Self {
        Self::yoloe_26_seg_tp()
            .with_version(Version::from(26))
            .with_scale(Scale::N)
            .with_model_file("yoloe-26n-seg-prompt.onnx")
    }

    /// YOLOE 26 small text-prompt segmentation model
    pub fn yoloe_26s_seg_tp() -> Self {
        Self::yoloe_26_seg_tp()
            .with_version(Version::from(26))
            .with_scale(Scale::S)
            .with_model_file("yoloe-26s-seg-prompt.onnx")
    }

    /// YOLOE 26 medium text-prompt segmentation model
    pub fn yoloe_26m_seg_tp() -> Self {
        Self::yoloe_26_seg_tp()
            .with_version(Version::from(26))
            .with_scale(Scale::M)
            .with_model_file("yoloe-26m-seg-prompt.onnx")
    }

    /// YOLOE 26 large text-prompt segmentation model
    pub fn yoloe_26l_seg_tp() -> Self {
        Self::yoloe_26_seg_tp()
            .with_version(Version::from(26))
            .with_scale(Scale::L)
            .with_model_file("yoloe-26l-seg-prompt.onnx")
    }

    /// YOLOE 26 extra large text-prompt segmentation model
    pub fn yoloe_26x_seg_tp() -> Self {
        Self::yoloe_26_seg_tp()
            .with_version(Version::from(26))
            .with_scale(Scale::X)
            .with_model_file("yoloe-26x-seg-prompt.onnx")
    }

    /// Base configuration for YOLOE visual-prompt segmentation
    fn yoloe_seg_vp() -> Self {
        Self::yoloe()
            .with_batch_size_all(1)
            .with_nc(80)
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 640)
            .with_model_ixx(0, 3, 640)
            .with_model_ixx(1, 1, (1, 80, 300)) // max nc
            .with_visual_encoder_ixx(0, 0, 1)
            .with_visual_encoder_ixx(0, 1, 3)
            .with_visual_encoder_ixx(0, 2, 640) // h
            .with_visual_encoder_ixx(0, 3, 640) // w
            .with_visual_encoder_ixx(1, 0, 1)
            .with_visual_encoder_ixx(1, 1, (1, 80, 300))
            .with_visual_encoder_ixx(1, 2, 80) // 1 / 8 * h
            .with_visual_encoder_ixx(1, 3, 80) // 1 / 8 * w
    }

    /// YOLOE v8 small visual-prompt segmentation model
    pub fn yoloe_v8s_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(Version::from(8))
            .with_scale(Scale::S)
            .with_visual_encoder_file("yoloe-v8s-savpe.onnx")
            .with_model_file("yoloe-v8s-seg-prompt.onnx")
    }

    /// YOLOE v8 medium visual-prompt segmentation model
    pub fn yoloe_v8m_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(Version::from(8))
            .with_scale(Scale::M)
            .with_visual_encoder_file("yoloe-v8m-savpe.onnx")
            .with_model_file("yoloe-v8m-seg-prompt.onnx")
    }

    /// YOLOE v8 large visual-prompt segmentation model
    pub fn yoloe_v8l_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(Version::from(8))
            .with_scale(Scale::L)
            .with_visual_encoder_file("yoloe-v8l-savpe.onnx")
            .with_model_file("yoloe-v8l-seg-prompt.onnx")
    }

    /// YOLOE 11 small visual-prompt segmentation model
    pub fn yoloe_11s_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(Version::from(11))
            .with_scale(Scale::S)
            .with_visual_encoder_file("yoloe-11s-savpe.onnx")
            .with_model_file("yoloe-11s-seg-prompt.onnx")
    }

    /// YOLOE 11 medium visual-prompt segmentation model
    pub fn yoloe_11m_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(Version::from(11))
            .with_scale(Scale::M)
            .with_visual_encoder_file("yoloe-11m-savpe.onnx")
            .with_model_file("yoloe-11m-seg-prompt.onnx")
    }

    /// YOLOE 11 large visual-prompt segmentation model
    pub fn yoloe_11l_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(Version::from(11))
            .with_scale(Scale::L)
            .with_visual_encoder_file("yoloe-11l-savpe.onnx")
            .with_model_file("yoloe-11l-seg-prompt.onnx")
    }

    /// YOLOE 26 small visual-prompt segmentation model
    pub fn yoloe_26n_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(Version::from(26))
            .with_scale(Scale::N)
            .with_visual_encoder_file("yoloe-26n-savpe.onnx")
            .with_model_file("yoloe-26n-seg-prompt.onnx")
    }

    /// YOLOE 26 medium visual-prompt segmentation model
    pub fn yoloe_26s_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(Version::from(26))
            .with_scale(Scale::S)
            .with_visual_encoder_file("yoloe-26s-savpe.onnx")
            .with_model_file("yoloe-26s-seg-prompt.onnx")
    }

    /// YOLOE 26 medium visual-prompt segmentation model
    pub fn yoloe_26m_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(Version::from(26))
            .with_scale(Scale::M)
            .with_visual_encoder_file("yoloe-26m-savpe.onnx")
            .with_model_file("yoloe-26m-seg-prompt.onnx")
    }

    /// YOLOE 26 large visual-prompt segmentation model
    pub fn yoloe_26l_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(Version::from(26))
            .with_scale(Scale::L)
            .with_visual_encoder_file("yoloe-26l-savpe.onnx")
            .with_model_file("yoloe-26l-seg-prompt.onnx")
    }

    /// YOLOE 26 extra large visual-prompt segmentation model
    pub fn yoloe_26x_seg_vp() -> Self {
        Self::yoloe_seg_vp()
            .with_version(Version::from(26))
            .with_scale(Scale::X)
            .with_visual_encoder_file("yoloe-26x-savpe.onnx")
            .with_model_file("yoloe-26x-seg-prompt.onnx")
    }
}
