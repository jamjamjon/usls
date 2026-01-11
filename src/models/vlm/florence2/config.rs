///
/// > # Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks
/// >
/// > Unified vision-language model capable of handling diverse vision tasks with a single architecture.
/// >
/// > # Paper & Code
/// >
/// > - **Hugging Face**: [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base)
/// > - **Paper**: [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242)
/// >
/// > # Model Variants
/// >
/// > - **florence2-base**: Base model with 768x768 input resolution
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Unified Vision Tasks**: Single model for multiple vision tasks
/// > - [X] **Multi-Modal Understanding**: Image and text processing
/// > - [X] **High-Resolution Input**: 768x768 image processing
/// > - [X] **Flexible Architecture**: Encoder-decoder structure
/// >
/// Model configuration for `Florence2`
///
impl crate::Config {
    /// Base configuration for Florence2 models
    pub fn florence2() -> Self {
        Self::default()
            .with_name("florence2")
            .with_batch_size_all(1)
            .with_visual_ixx(0, 1, 3)
            .with_visual_ixx(0, 2, 768)
            .with_visual_ixx(0, 3, 768)
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
    }

    /// Base model with 768x768 input resolution
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
}
