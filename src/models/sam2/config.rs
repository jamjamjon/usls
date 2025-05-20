use crate::Config;

/// Model configuration for `SAM2.1`
impl Config {
    pub fn sam2_1_tiny() -> Self {
        Self::sam()
            .with_encoder_file("sam2.1-hiera-tiny-encoder.onnx")
            .with_decoder_file("sam2.1-hiera-tiny-decoder.onnx")
    }

    pub fn sam2_1_small() -> Self {
        Self::sam()
            .with_encoder_file("sam2.1-hiera-small-encoder.onnx")
            .with_decoder_file("sam2.1-hiera-small-decoder.onnx")
    }

    pub fn sam2_1_base_plus() -> Self {
        Self::sam()
            .with_encoder_file("sam2.1-hiera-base-plus-encoder.onnx")
            .with_decoder_file("sam2.1-hiera-base-plus-decoder.onnx")
    }

    pub fn sam2_1_large() -> Self {
        Self::sam()
            .with_encoder_file("sam2.1-hiera-large-encoder.onnx")
            .with_decoder_file("sam2.1-hiera-large-decoder.onnx")
    }
}
