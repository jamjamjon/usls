/// Model configuration for `SmolVLM`
impl crate::Config {
    pub fn smolvlm() -> Self {
        Self::default()
            .with_name("smolvlm")
            .with_batch_size_all(1)
            .with_image_mean(&[0.5, 0.5, 0.5])
            .with_image_std(&[0.5, 0.5, 0.5])
            .with_resize_filter("lanczos3")
            .with_tokenizer_file("smolvlm/tokenizer.json")
    }

    pub fn smolvlm_256m() -> Self {
        Self::smolvlm()
            .with_scale(crate::Scale::Million(256.))
            .with_visual_file("256m-vision-encoder.onnx")
            .with_textual_file("256m-embed-tokens.onnx")
            .with_textual_decoder_merged_file("256m-decoder-model-merged.onnx")
    }

    pub fn smolvlm_500m() -> Self {
        Self::smolvlm()
            .with_scale(crate::Scale::Million(500.))
            .with_visual_file("500m-vision-encoder.onnx")
            .with_textual_file("500m-embed-tokens.onnx")
            .with_textual_decoder_merged_file("500m-decoder-model-merged.onnx")
    }
}
