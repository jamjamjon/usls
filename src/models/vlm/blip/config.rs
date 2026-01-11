///
/// > # BLIP: Bootstrapping Language-Image Pre-training
/// >
/// > Unified vision-language understanding and generation model with bootstrapping pre-training.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [salesforce/BLIP](https://github.com/salesforce/BLIP)
/// > - **Paper**: [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)
/// >
/// > # Model Variants
/// >
/// > - **blip-v1-base-caption**: BLIP v1 base model for image captioning
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Image-Text Captioning**: Generate descriptions for images
/// > - [ ] **Visual Question Answering**: Answer questions about images
/// > - [ ] **Image-Text Retrieval**: Retrieve images or text based on queries
/// > - [ ] **TensorRT Support**: Optimized inference for textual models
/// >
/// Model configuration for `BLIP`
///
impl crate::Config {
    /// Base configuration for BLIP models
    #[allow(clippy::excessive_precision)]
    pub fn blip() -> Self {
        Self::default()
            .with_name("blip")
            .with_batch_size_all(1)
            .with_visual_ixx(0, 1, 3)
            .with_visual_ixx(0, 2, 384)
            .with_visual_ixx(0, 3, 384)
            .with_image_mean([0.48145466, 0.4578275, 0.40821073])
            .with_image_std([0.26862954, 0.26130258, 0.27577711])
    }

    /// BLIP v1 base model for image captioning
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
