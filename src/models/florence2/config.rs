/// Model configuration for `Florence2`
impl crate::Options {
    pub fn florence2() -> Self {
        Self::default()
            .with_model_name("florence2")
            .with_batch_size(1)
    }

    pub fn florence2_visual() -> Self {
        Self::florence2()
            .with_model_kind(crate::Kind::Vision)
            .with_model_ixx(0, 2, 768.into())
            .with_model_ixx(0, 3, 768.into())
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_resize_filter("Bilinear")
            .with_normalize(true)
    }

    pub fn florence2_textual() -> Self {
        Self::florence2().with_model_kind(crate::Kind::Language)
    }

    pub fn florence2_visual_base() -> Self {
        Self::florence2_visual().with_model_scale(crate::Scale::B)
    }

    pub fn florence2_textual_base() -> Self {
        Self::florence2_textual().with_model_scale(crate::Scale::B)
    }

    pub fn florence2_visual_large() -> Self {
        Self::florence2_visual().with_model_scale(crate::Scale::L)
    }

    pub fn florence2_textual_large() -> Self {
        Self::florence2_textual().with_model_scale(crate::Scale::L)
    }

    pub fn florence2_visual_encoder_base() -> Self {
        Self::florence2_visual_base().with_model_file("base-vision-encoder.onnx")
    }

    pub fn florence2_textual_embed_base() -> Self {
        Self::florence2_textual_base().with_model_file("base-embed-tokens.onnx")
    }

    pub fn florence2_texual_encoder_base() -> Self {
        Self::florence2_textual_base().with_model_file("base-encoder.onnx")
    }

    pub fn florence2_texual_decoder_base() -> Self {
        Self::florence2_textual_base().with_model_file("base-decoder.onnx")
    }

    pub fn florence2_texual_decoder_merged_base() -> Self {
        Self::florence2_textual_base().with_model_file("base-decoder-merged.onnx")
    }
}
