/// Model configuration for `SmolVLM`
impl crate::Options {
    pub fn smolvlm() -> Self {
        Self::default()
            .with_batch_size(1)
            .with_model_name("smolvlm")
            .with_model_num_dry_run(3)
    }

    pub fn smolvlm_vision() -> Self {
        Self::smolvlm()
            .with_model_kind(crate::Kind::Vision)
            .with_image_mean(&[0.5, 0.5, 0.5])
            .with_image_std(&[0.5, 0.5, 0.5])
            .with_resize_filter("lanczos3")
            .with_normalize(true)
    }

    pub fn smolvlm_text() -> Self {
        Self::smolvlm().with_model_kind(crate::Kind::Language)
    }

    pub fn smolvlm_vision_256m() -> Self {
        Self::smolvlm_vision()
            .with_model_scale(crate::Scale::Million(256.))
            .with_model_file("256m-vision-encoder.onnx")
    }

    pub fn smolvlm_text_embed_256m() -> Self {
        Self::smolvlm_text()
            .with_model_scale(crate::Scale::Million(256.))
            .with_model_file("256m-embed-tokens.onnx")
    }

    pub fn smolvlm_decoder_256m() -> Self {
        Self::smolvlm_text()
            .with_model_scale(crate::Scale::Million(256.))
            .with_model_file("256m-decoder-model-merged.onnx")
    }

    pub fn smolvlm_vision_500m() -> Self {
        Self::smolvlm_vision()
            .with_model_scale(crate::Scale::Million(500.))
            .with_model_file("500m-vision-encoder.onnx")
    }

    pub fn smolvlm_text_embed_500m() -> Self {
        Self::smolvlm_text()
            .with_model_scale(crate::Scale::Million(500.))
            .with_model_file("500m-embed-tokens.onnx")
    }

    pub fn smolvlm_decoder_500m() -> Self {
        Self::smolvlm_text()
            .with_model_scale(crate::Scale::Million(500.))
            .with_model_file("500m-decoder-model-merged.onnx")
    }
}
