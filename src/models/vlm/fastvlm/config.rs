///
/// > # FastVLM: Efficient Vision Encoding for Vision Language Models
/// >
/// > Efficient vision encoding architecture designed for high-performance vision-language models.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [apple/ml-fastvlm](https://github.com/apple/ml-fastvlm)
/// > - **ONNX Models**: [FastVLM-0.5B-ONNX](https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX)
/// >
/// > # Model Variants
/// >
/// > - **fastvlm-0.5b**: FastVLM 0.5 billion parameter model
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Efficient Vision Encoding**: Optimized visual feature extraction
/// > - [X] **High-Resolution Input**: 1024x1024 image processing
/// > - [X] **Lightweight Architecture**: 0.5B parameters for efficient inference
/// > - [X] **ONNX Optimization**: Optimized for ONNX runtime
/// >
/// Model configuration for `FastVLM`
///
impl crate::Config {
    /// Base configuration for FastVLM models
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

    /// FastVLM 0.5 billion parameter model
    pub fn fastvlm_0_5b() -> Self {
        Self::fastvlm()
            .with_scale(crate::Scale::Billion(0.5))
            .with_visual_file("0.5b-vision-encoder.onnx")
            .with_textual_file("0.5b-embed-tokens.onnx")
            .with_textual_decoder_merged_file("0.5b-decoder-model-merged.onnx")
    }
}
