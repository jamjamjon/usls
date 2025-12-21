use crate::Config;

/// Model configuration for `SAM2.1`
impl Config {
    /// Creates a configuration for SAM 2.1 tiny model.
    ///
    /// The smallest variant of the hierarchical architecture, optimized for speed.
    pub fn sam2_1_tiny() -> Self {
        Self::sam()
            .with_encoder_file("sam2.1-hiera-tiny-encoder.onnx")
            .with_decoder_file("sam2.1-hiera-tiny-decoder.onnx")
    }

    /// Creates a configuration for SAM 2.1 small model.
    ///
    /// A balanced variant offering good performance and efficiency.
    pub fn sam2_1_small() -> Self {
        Self::sam()
            .with_encoder_file("sam2.1-hiera-small-encoder.onnx")
            .with_decoder_file("sam2.1-hiera-small-decoder.onnx")
    }

    /// Creates a configuration for SAM 2.1 base-plus model.
    ///
    /// An enhanced base model with improved segmentation quality.
    pub fn sam2_1_base_plus() -> Self {
        Self::sam()
            .with_encoder_file("sam2.1-hiera-base-plus-encoder.onnx")
            .with_decoder_file("sam2.1-hiera-base-plus-decoder.onnx")
    }

    /// Creates a configuration for SAM 2.1 large model.
    ///
    /// The most powerful variant with highest segmentation quality.
    pub fn sam2_1_large() -> Self {
        Self::sam()
            .with_encoder_file("sam2.1-hiera-large-encoder.onnx")
            .with_decoder_file("sam2.1-hiera-large-decoder.onnx")
    }
}
