use crate::Scale;

///
/// > # TrOCR: Transformer-based Optical Character Recognition
/// >
/// > Transformer-based OCR model with pre-trained models for both printed and handwritten text recognition.
/// >
/// > # Paper & Code
/// >
/// > - **Hugging Face**: [microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed)
/// > - **Paper**: [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282)
/// >
/// > # Model Variants
/// >
/// > - **trocr-small-printed**: Small model optimized for printed text
/// > - **trocr-base-printed**: Base model optimized for printed text
/// > - **trocr-small-handwritten**: Small model optimized for handwritten text
/// > - **trocr-base-handwritten**: Base model optimized for handwritten text
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Printed Text Recognition**: High accuracy on printed documents
/// > - [X] **Handwritten Text Recognition**: Robust performance on handwriting
/// > - [X] **Transformer Architecture**: Modern transformer-based OCR
/// > - [X] **Multi-Scale Support**: Small and base model variants
/// >
/// Model configuration for `TrOCR`
///
impl crate::Config {
    /// Base configuration for TrOCR models with default settings
    pub fn trocr() -> Self {
        Self::default()
            .with_name("trocr")
            .with_batch_size_all(1)
            .with_visual_ixx(0, 1, 3)
            .with_visual_ixx(0, 2, 384)
            .with_visual_ixx(0, 3, 384)
            .with_image_mean([0.5, 0.5, 0.5])
            .with_image_std([0.5, 0.5, 0.5])
            .with_resize_filter(crate::ResizeFilter::Lanczos3)
            .with_tokenizer_file("trocr/tokenizer.json")
            .with_config_file("trocr/config.json")
            .with_special_tokens_map_file("trocr/special_tokens_map.json")
            .with_tokenizer_config_file("trocr/tokenizer_config.json")
    }

    /// Small model optimized for printed text
    pub fn trocr_small_printed() -> Self {
        Self::trocr()
            .with_scale(Scale::S)
            .with_visual_file("s-encoder-printed.onnx")
            .with_textual_decoder_file("s-decoder-printed.onnx")
            .with_textual_decoder_merged_file("s-decoder-merged-printed.onnx")
            .with_tokenizer_file("trocr/tokenizer-small.json")
    }

    /// Base model optimized for handwritten text
    pub fn trocr_base_handwritten() -> Self {
        Self::trocr()
            .with_scale(Scale::B)
            .with_visual_file("b-encoder-handwritten.onnx")
            .with_textual_decoder_file("b-decoder-handwritten.onnx")
            .with_textual_decoder_merged_file("b-decoder-merged-handwritten.onnx")
            .with_tokenizer_file("trocr/tokenizer-base.json")
    }

    /// Small model optimized for handwritten text
    pub fn trocr_small_handwritten() -> Self {
        Self::trocr_small_printed()
            .with_visual_file("s-encoder-handwritten.onnx")
            .with_textual_decoder_file("s-decoder-handwritten.onnx")
            .with_textual_decoder_merged_file("s-decoder-merged-handwritten.onnx")
    }

    /// Base model optimized for printed text
    pub fn trocr_base_printed() -> Self {
        Self::trocr_base_handwritten()
            .with_visual_file("b-encoder-printed.onnx")
            .with_textual_decoder_file("b-decoder-printed.onnx")
            .with_textual_decoder_merged_file("b-decoder-merged-printed.onnx")
    }
}
