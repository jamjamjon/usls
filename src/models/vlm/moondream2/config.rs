///
/// > # MoonDream2: Tiny Vision Language Model That Kicks Ass
/// >
/// > Compact yet powerful vision-language model designed to run efficiently anywhere.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [vikhyat/moondream](https://github.com/vikhyat/moondream/tree/main)
/// >
/// > # Model Variants
/// >
/// > - **moondream2-0.5b**: 0.5 billion parameter model
/// > - **moondream2-2b**: 2 billion parameter model
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Compact Architecture**: Efficient small-scale models
/// > - [X] **Multi-Component Design**: Separate encoders for vision, text, coordinates, and size
/// > - [X] **Universal Deployment**: Runs efficiently on various hardware
/// > - [X] **Visual Understanding**: Comprehensive image-text understanding
/// >
/// Model configuration for `moondream2`
///
impl crate::Config {
    /// Base configuration for MoonDream2 models
    pub fn moondream2() -> Self {
        Self::default()
            .with_name("moondream2")
            .with_visual_encoder_ixx(0, 0, (1, 3, 4)) // patch count
            .with_image_mean([0.5, 0.5, 0.5])
            .with_image_std([0.5, 0.5, 0.5])
            .with_resize_mode_type(crate::ResizeModeType::FitExact)
            .with_resize_filter(crate::ResizeFilter::CatmullRom)
            .with_visual_projection_ixx(0, 0, 1)
            .with_textual_encoder_ixx(0, 0, 1)
            .with_textual_decoder_ixx(0, 0, 1)
            .with_size_encoder_ixx(0, 0, 1)
            .with_size_decoder_ixx(0, 0, 1)
            .with_coord_encoder_ixx(0, 0, 1)
            .with_coord_decoder_ixx(0, 0, 1)
            .with_tokenizer_file("moondream2/tokenizer.json")
            .with_tokenizer_config_file("moondream2/tokenizer_config.json")
    }

    /// 0.5 billion parameter model
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

    /// 2 billion parameter model
    pub fn moondream2_2b() -> Self {
        Self::moondream2()
            .with_scale(crate::Scale::Billion(2.))
            .with_visual_encoder_file("2b-vision-encoder.onnx")
            .with_visual_projection_file("2b-vision-projection.onnx")
            .with_textual_decoder_file("2b-text-decoder.onnx")
            .with_textual_encoder_file("2b-text-encoder.onnx")
            .with_size_encoder_file("2b-size-encoder.onnx")
            .with_size_decoder_file("2b-size-decoder.onnx")
    }
}
