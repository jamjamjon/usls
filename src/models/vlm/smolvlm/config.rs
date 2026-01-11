///
/// > # SmolVLM: Small Yet Mighty Vision Language Model
/// >
/// > Compact vision-language models designed for efficient deployment while maintaining strong performance.
/// >
/// > # Paper & Code
/// >
/// > - **SmolVLM-256M**: [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
/// > - **SmolVLM-500M**: [HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
/// >
/// > # Model Variants
/// >
/// > - **smolvlm-256m**: SmolVLM v1 with 256 million parameters
/// > - **smolvlm-500m**: SmolVLM v1 with 500 million parameters
/// > - **smolvlm2-256m**: SmolVLM v2 with 256 million parameters
/// > - **smolvlm2-500m**: SmolVLM v2 with 500 million parameters
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Compact Architecture**: Small parameter count for efficient deployment
/// > - [X] **Instruction Following**: Optimized for instruction-based tasks
/// > - [X] **High-Quality Vision**: Lanczos3 filter for superior image processing
/// > - [X] **Multi-Generation**: Both v1 and v2 architectures available
/// >
/// Model configuration for `SmolVLM`
///
impl crate::Config {
    /// Base configuration for SmolVLM v1 models
    pub fn smolvlm() -> Self {
        Self::default()
            .with_name("smolvlm")
            .with_version(1.into())
            .with_batch_size_all(1)
            .with_image_mean([0.5, 0.5, 0.5])
            .with_image_std([0.5, 0.5, 0.5])
            .with_resize_filter(crate::ResizeFilter::Lanczos3)
            .with_tokenizer_file("smolvlm/tokenizer.json")
            .with_tokenizer_config_file("smolvlm/tokenizer_config.json")
    }

    /// SmolVLM v1 with 256 million parameters
    pub fn smolvlm_256m() -> Self {
        Self::smolvlm()
            .with_scale(crate::Scale::Million(256.))
            .with_visual_file("256m-vision-encoder.onnx")
            .with_textual_file("256m-embed-tokens.onnx")
            .with_textual_decoder_merged_file("256m-decoder-model-merged.onnx")
    }

    /// SmolVLM v1 with 500 million parameters
    pub fn smolvlm_500m() -> Self {
        Self::smolvlm()
            .with_scale(crate::Scale::Million(500.))
            .with_visual_file("500m-vision-encoder.onnx")
            .with_textual_file("500m-embed-tokens.onnx")
            .with_textual_decoder_merged_file("500m-decoder-model-merged.onnx")
    }

    /// Base configuration for SmolVLM v2 models
    pub fn smolvlm2() -> Self {
        Self::default()
            .with_name("smolvlm2")
            .with_version(2.into())
            .with_batch_size_all(1)
            .with_image_mean([0.5, 0.5, 0.5])
            .with_image_std([0.5, 0.5, 0.5])
            .with_resize_filter(crate::ResizeFilter::Lanczos3)
            .with_tokenizer_file("smolvlm2/tokenizer.json")
            .with_tokenizer_config_file("smolvlm2/tokenizer_config.json")
    }

    /// SmolVLM v2 with 256 million parameters
    pub fn smolvlm2_256m() -> Self {
        Self::smolvlm2()
            .with_scale(crate::Scale::Million(256.))
            .with_visual_file("256m-vision-encoder.onnx")
            .with_textual_file("256m-embed-tokens.onnx")
            .with_textual_decoder_merged_file("256m-decoder-model-merged.onnx")
    }

    /// SmolVLM v2 with 500 million parameters
    pub fn smolvlm2_500m() -> Self {
        Self::smolvlm2()
            .with_scale(crate::Scale::Million(500.))
            .with_visual_file("500m-vision-encoder.onnx")
            .with_textual_file("500m-embed-tokens.onnx")
            .with_textual_decoder_merged_file("500m-decoder-model-merged.onnx")
    }
}
