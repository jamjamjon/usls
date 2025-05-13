use crate::Options;

/// Model configuration for `SAM2.1`
impl Options {
    pub fn sam2_encoder() -> Self {
        Self::sam()
            .with_model_ixx(0, 2, 1024.into())
            .with_model_ixx(0, 3, 1024.into())
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_resize_filter("Bilinear")
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
    }

    pub fn sam2_decoder() -> Self {
        Self::sam()
    }

    pub fn sam2_1_tiny_encoder() -> Self {
        Self::sam2_encoder().with_model_file("sam2.1-hiera-tiny-encoder.onnx")
    }

    pub fn sam2_1_tiny_decoder() -> Self {
        Self::sam2_decoder().with_model_file("sam2.1-hiera-tiny-decoder.onnx")
    }

    pub fn sam2_1_small_encoder() -> Self {
        Self::sam2_encoder().with_model_file("sam2.1-hiera-small-encoder.onnx")
    }

    pub fn sam2_1_small_decoder() -> Self {
        Self::sam2_decoder().with_model_file("sam2.1-hiera-small-decoder.onnx")
    }

    pub fn sam2_1_base_plus_encoder() -> Self {
        Self::sam2_encoder().with_model_file("sam2.1-hiera-base-plus-encoder.onnx")
    }

    pub fn sam2_1_base_plus_decoder() -> Self {
        Self::sam2_decoder().with_model_file("sam2.1-hiera-base-plus-decoder.onnx")
    }

    pub fn sam2_1_large_encoder() -> Self {
        Self::sam2_encoder().with_model_file("sam2.1-hiera-large-encoder.onnx")
    }

    pub fn sam2_1_large_decoder() -> Self {
        Self::sam2_decoder().with_model_file("sam2.1-hiera-large-decoder.onnx")
    }
}
