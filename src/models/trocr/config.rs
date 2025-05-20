use crate::Scale;

/// Model configuration for `TrOCR`
impl crate::Config {
    pub fn trocr() -> Self {
        Self::default()
            .with_name("trocr")
            .with_batch_size_all(1)
            .with_visual_ixx(0, 1, 3.into())
            .with_visual_ixx(0, 2, 384.into())
            .with_visual_ixx(0, 3, 384.into())
            .with_image_mean(&[0.5, 0.5, 0.5])
            .with_image_std(&[0.5, 0.5, 0.5])
            .with_resize_filter("lanczos3")
            .with_tokenizer_file("trocr/tokenizer.json")
            .with_config_file("trocr/config.json")
            .with_special_tokens_map_file("trocr/special_tokens_map.json")
            .with_tokenizer_config_file("trocr/tokenizer_config.json")
    }

    pub fn trocr_small_printed() -> Self {
        Self::trocr()
            .with_scale(Scale::S)
            .with_visual_file("s-encoder-printed.onnx")
            .with_textual_decoder_file("s-decoder-printed.onnx")
            .with_textual_decoder_merged_file("s-decoder-merged-printed.onnx")
            .with_tokenizer_file("trocr/tokenizer-small.json")
    }

    pub fn trocr_base_handwritten() -> Self {
        Self::trocr()
            .with_scale(Scale::B)
            .with_visual_file("b-encoder-handwritten.onnx")
            .with_textual_decoder_file("b-decoder-handwritten.onnx")
            .with_textual_decoder_merged_file("b-decoder-merged-handwritten.onnx")
            .with_tokenizer_file("trocr/tokenizer-base.json")
    }

    pub fn trocr_small_handwritten() -> Self {
        Self::trocr_small_printed()
            .with_visual_file("s-encoder-handwritten.onnx")
            .with_textual_decoder_file("s-decoder-handwritten.onnx")
            .with_textual_decoder_merged_file("s-decoder-merged-handwritten.onnx")
    }

    pub fn trocr_base_printed() -> Self {
        Self::trocr_base_handwritten()
            .with_visual_file("b-encoder-printed.onnx")
            .with_textual_decoder_file("b-decoder-printed.onnx")
            .with_textual_decoder_merged_file("b-decoder-merged-printed.onnx")
    }
}
