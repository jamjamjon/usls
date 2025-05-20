/// Model configuration for `BLIP`
impl crate::Config {
    #[allow(clippy::excessive_precision)]
    pub fn blip() -> Self {
        Self::default()
            .with_name("blip")
            .with_batch_size_all(1)
            .with_visual_ixx(0, 1, 3.into())
            .with_visual_ixx(0, 2, 384.into())
            .with_visual_ixx(0, 3, 384.into())
            .with_image_mean(&[0.48145466, 0.4578275, 0.40821073])
            .with_image_std(&[0.26862954, 0.26130258, 0.27577711])
    }

    pub fn blip_v1_base_caption() -> Self {
        Self::blip()
            .with_version(1.into())
            .with_visual_file("v1-base-caption-visual.onnx")
            .with_textual_file("v1-base-caption-textual.onnx")
            .with_tokenizer_file("blip/tokenizer.json")
            .with_tokenizer_config_file("blip/tokenizer_config.json")
            .with_special_tokens_map_file("blip/special_tokens_map.json")
    }
}
