/// Model configuration for `moondream2`
impl crate::Config {
    pub fn moondream2() -> Self {
        Self::default()
            .with_name("moondream2")
            .with_visual_encoder_ixx(0, 0, (1, 3, 4).into()) // patch count
            .with_image_mean(&[0.5, 0.5, 0.5])
            .with_image_std(&[0.5, 0.5, 0.5])
            .with_resize_mode(crate::ResizeMode::FitExact)
            .with_resize_filter("catmullrom")
            .with_visual_projection_ixx(0, 0, 1.into())
            .with_textual_encoder_ixx(0, 0, 1.into())
            .with_textual_decoder_ixx(0, 0, 1.into())
            .with_size_encoder_ixx(0, 0, 1.into())
            .with_size_decoder_ixx(0, 0, 1.into())
            .with_coord_encoder_ixx(0, 0, 1.into())
            .with_coord_decoder_ixx(0, 0, 1.into())
            .with_tokenizer_file("moondream2/tokenizer.json")
            .with_tokenizer_config_file("moondream2/tokenizer_config.json")
    }

    pub fn moondream2_0_5b() -> Self {
        Self::moondream2()
            .with_scale(crate::Scale::Billion(0.5))
            .with_visual_encoder_file("0.5b-vision-encoder.onnx")
            .with_visual_projection_file("0.5b-vision-projection.onnx")
            .with_textual_decoder_file("0.5b-text-decoder.onnx")
            .with_textual_encoder_file("0.5b-text-encoder.onnx")
            .with_coord_encoder_file("0.5b-coord-encoder.onnx")
            .with_coord_decoder_file("0.5b-coord-decoder.onnx")
            .with_size_encoder_file("0.5b-size-encoder.onnx")
            .with_size_decoder_file("0.5b-size-decoder.onnx")
    }

    pub fn moondream2_2b() -> Self {
        Self::moondream2()
            .with_scale(crate::Scale::Billion(2.))
            .with_visual_encoder_file("2b-vision-encoder.onnx")
            .with_visual_projection_file("2b-vision-projection.onnx")
            .with_textual_decoder_file("2b-text-decoder.onnx")
            .with_textual_encoder_file("2b-text-encoder.onnx")
            .with_coord_encoder_file("2b-coord-encoder.onnx")
            .with_coord_decoder_file("2b-coord-decoder.onnx")
            .with_size_encoder_file("2b-size-encoder.onnx")
            .with_size_decoder_file("2b-size-decoder.onnx")
    }
}
