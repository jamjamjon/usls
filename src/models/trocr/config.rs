use crate::Scale;

/// Model configuration for `TrOCR`
impl crate::Options {
    pub fn trocr() -> Self {
        Self::default().with_model_name("trocr").with_batch_size(1)
    }

    pub fn trocr_visual() -> Self {
        Self::trocr()
            .with_model_kind(crate::Kind::Vision)
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 384.into())
            .with_model_ixx(0, 3, 384.into())
            .with_image_mean(&[0.5, 0.5, 0.5])
            .with_image_std(&[0.5, 0.5, 0.5])
            .with_resize_filter("Bilinear")
            .with_normalize(true)
    }

    pub fn trocr_textual() -> Self {
        Self::trocr().with_model_kind(crate::Kind::Language)
    }

    pub fn trocr_visual_small() -> Self {
        Self::trocr_visual().with_model_scale(Scale::S)
    }

    pub fn trocr_textual_small() -> Self {
        Self::trocr_textual()
            .with_model_scale(Scale::S)
            .with_tokenizer_file("trocr/tokenizer-small.json")
    }

    pub fn trocr_visual_base() -> Self {
        Self::trocr_visual().with_model_scale(Scale::B)
    }

    pub fn trocr_textual_base() -> Self {
        Self::trocr_textual()
            .with_model_scale(Scale::B)
            .with_tokenizer_file("trocr/tokenizer-base.json")
    }

    pub fn trocr_encoder_small_printed() -> Self {
        Self::trocr_visual_small().with_model_file("s-encoder-printed.onnx")
    }

    pub fn trocr_decoder_small_printed() -> Self {
        Self::trocr_textual_small().with_model_file("s-decoder-printed.onnx")
    }

    pub fn trocr_decoder_merged_small_printed() -> Self {
        Self::trocr_textual_small().with_model_file("s-decoder-merged-printed.onnx")
    }

    pub fn trocr_encoder_small_handwritten() -> Self {
        Self::trocr_visual_small().with_model_file("s-encoder-handwritten.onnx")
    }

    pub fn trocr_decoder_small_handwritten() -> Self {
        Self::trocr_textual_small().with_model_file("s-decoder-handwritten.onnx")
    }

    pub fn trocr_decoder_merged_small_handwritten() -> Self {
        Self::trocr_textual_small().with_model_file("s-decoder-merged-handwritten.onnx")
    }

    pub fn trocr_encoder_base_printed() -> Self {
        Self::trocr_visual_base().with_model_file("b-encoder-printed.onnx")
    }

    pub fn trocr_decoder_base_printed() -> Self {
        Self::trocr_textual_base().with_model_file("b-decoder-printed.onnx")
    }

    pub fn trocr_decoder_merged_base_printed() -> Self {
        Self::trocr_textual_base().with_model_file("b-decoder-merged-printed.onnx")
    }

    pub fn trocr_encoder_base_handwritten() -> Self {
        Self::trocr_visual_base().with_model_file("b-encoder-handwritten.onnx")
    }

    pub fn trocr_decoder_base_handwritten() -> Self {
        Self::trocr_textual_base().with_model_file("b-decoder-handwritten.onnx")
    }

    pub fn trocr_decoder_merged_base_handwritten() -> Self {
        Self::trocr_textual_base().with_model_file("b-decoder-merged-handwritten.onnx")
    }
}
