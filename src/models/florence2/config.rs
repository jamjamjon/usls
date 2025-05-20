/// Model configuration for `Florence2`
impl crate::Config {
    pub fn florence2() -> Self {
        Self::default()
            .with_name("florence2")
            .with_batch_size_all(1)
            .with_visual_ixx(0, 1, 3.into())
            .with_visual_ixx(0, 2, 768.into())
            .with_visual_ixx(0, 3, 768.into())
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
    }

    pub fn florence2_base() -> Self {
        Self::florence2()
            .with_scale(crate::Scale::B)
            .with_visual_file("base-vision-encoder.onnx")
            .with_textual_file("base-embed-tokens.onnx")
            .with_textual_encoder_file("base-encoder.onnx")
            .with_textual_decoder_file("base-decoder.onnx")
            .with_textual_decoder_merged_file("base-decoder-merged.onnx")
            .with_tokenizer_file("florence2/tokenizer.json")
            .with_config_file("florence2/config.json")
            .with_special_tokens_map_file("florence2/special_tokens_map.json")
            .with_tokenizer_config_file("florence2/tokenizer_config.json")
    }

    pub fn florence2_large() -> Self {
        todo!()
    }
}
