/// Model configuration for `FastVLM`
impl crate::Config {
    pub fn fastvlm() -> Self {
        Self::default()
            .with_name("fastvlm")
            .with_batch_size_all(1)
            .with_visual_ixx(0, 1, 3)
            .with_visual_ixx(0, 2, 1024)
            .with_visual_ixx(0, 3, 1024)
            // .with_image_mean([0., 0., 0.])
            // .with_image_std([1., 1., 1.])
            .with_normalize(true)
            .with_resize_filter(crate::ResizeFilter::Bilinear)
            .with_tokenizer_file("fastvlm/tokenizer.json")
            .with_tokenizer_config_file("fastvlm/tokenizer_config.json")
            .with_config_file("fastvlm/config.json")
    }

    pub fn fastvlm_0_5b() -> Self {
        Self::fastvlm()
            .with_scale(crate::Scale::Billion(0.5))
            .with_visual_file("0.5b-vision-encoder.onnx")
            .with_textual_file("0.5b-embed-tokens.onnx")
            .with_textual_decoder_merged_file("0.5b-decoder-model-merged.onnx")
    }
}
