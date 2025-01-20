/// Model configuration for `moondream2`
impl crate::Options {
    pub fn moondream2() -> Self {
        Self::default()
            .with_model_name("moondream2")
            .with_model_num_dry_run(0)
    }

    pub fn moondream2_0_5b() -> Self {
        Self::moondream2().with_model_scale(crate::Scale::Billion(0.5))
    }

    pub fn moondream2_0_5b_vision_encoder() -> Self {
        Self::moondream2_0_5b()
            .with_model_ixx(0, 0, (1, 3, 4).into()) // patch count
            .with_model_kind(crate::Kind::Vision)
            .with_image_mean(&[0.5, 0.5, 0.5])
            .with_image_std(&[0.5, 0.5, 0.5])
            .with_normalize(true)
            .with_resize_mode(crate::ResizeMode::FitExact)
            .with_resize_filter("catmullrom")
            .with_model_file("0.5b-vision-encoder.onnx")
    }

    pub fn moondream2_0_5b_vision_projection() -> Self {
        Self::moondream2_0_5b()
            .with_batch_size(1)
            .with_model_kind(crate::Kind::Vision)
            .with_model_file("0.5b-vision-projection.onnx")
    }

    pub fn moondream2_0_5b_text_decoder() -> Self {
        Self::moondream2_0_5b()
            .with_batch_size(1)
            .with_model_kind(crate::Kind::Language)
            .with_model_file("0.5b-text-decoder.onnx")
    }

    pub fn moondream2_0_5b_text_encoder() -> Self {
        Self::moondream2_0_5b()
            .with_batch_size(1)
            .with_model_kind(crate::Kind::Language)
            .with_model_file("0.5b-text-encoder.onnx")
    }

    pub fn moondream2_0_5b_coord_encoder() -> Self {
        Self::moondream2_0_5b()
            .with_batch_size(1)
            .with_model_file("0.5b-coord-encoder.onnx")
    }

    pub fn moondream2_0_5b_coord_decoder() -> Self {
        Self::moondream2_0_5b()
            .with_batch_size(1)
            .with_model_file("0.5b-coord-decoder.onnx")
    }

    pub fn moondream2_0_5b_size_encoder() -> Self {
        Self::moondream2_0_5b()
            .with_batch_size(1)
            .with_model_file("0.5b-size-encoder.onnx")
    }

    pub fn moondream2_0_5b_size_decoder() -> Self {
        Self::moondream2_0_5b()
            .with_batch_size(1)
            .with_model_file("0.5b-size-decoder.onnx")
    }

    pub fn moondream2_2b_vision_encoder() -> Self {
        Self::moondream2_0_5b_vision_encoder()
            .with_model_scale(crate::Scale::Billion(2.))
            .with_model_file("2b-vision-encoder.onnx")
    }

    pub fn moondream2_2b_vision_projection() -> Self {
        Self::moondream2_0_5b_vision_projection()
            .with_model_scale(crate::Scale::Billion(2.))
            .with_model_file("2b-vision-projection.onnx")
    }

    pub fn moondream2_2b_text_decoder() -> Self {
        Self::moondream2_0_5b_text_decoder()
            .with_model_scale(crate::Scale::Billion(2.))
            .with_model_file("2b-text-decoder.onnx")
    }

    pub fn moondream2_2b_text_encoder() -> Self {
        Self::moondream2_0_5b_text_encoder()
            .with_model_scale(crate::Scale::Billion(2.))
            .with_model_file("2b-text-encoder.onnx")
    }

    pub fn moondream2_2b_coord_encoder() -> Self {
        Self::moondream2_0_5b_coord_encoder()
            .with_model_scale(crate::Scale::Billion(2.))
            .with_model_file("2b-coord-encoder.onnx")
    }

    pub fn moondream2_2b_coord_decoder() -> Self {
        Self::moondream2_0_5b_coord_decoder()
            .with_model_scale(crate::Scale::Billion(2.))
            .with_model_file("2b-coord-decoder.onnx")
    }

    pub fn moondream2_2b_size_encoder() -> Self {
        Self::moondream2_0_5b_size_encoder()
            .with_model_scale(crate::Scale::Billion(2.))
            .with_model_file("2b-size-encoder.onnx")
    }

    pub fn moondream2_2b_size_decoder() -> Self {
        Self::moondream2_0_5b_size_decoder()
            .with_model_scale(crate::Scale::Billion(2.))
            .with_model_file("2b-size-decoder.onnx")
    }
}
