use crate::Config;

/// Model configuration for `SAM3`
impl Config {
    /// SAM3 base configuration
    ///
    /// - Input size: 1008x1008 (FitExact, no aspect ratio preserved)
    /// - Normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    /// - Tokenizer: CLIP BPE (max_length=32)
    /// - Confidence threshold: 0.5
    pub fn sam3() -> Self {
        Self::default()
            .with_name("sam3")
            .with_batch_size_all_min_opt_max(1, 1, 4)
            .with_visual_encoder_ixx(0, 1, 3.into())
            .with_visual_encoder_ixx(0, 2, 1008.into())
            .with_visual_encoder_ixx(0, 3, 1008.into())
            .with_textual_encoder_ixx(0, 1, 32.into())
            .with_resize_mode(crate::ResizeMode::FitExact)
            .with_resize_filter("Bilinear")
            .with_image_mean(&[0.5, 0.5, 0.5])
            .with_image_std(&[0.5, 0.5, 0.5])
            .with_normalize(true)
            .with_find_contours(true)
            .with_class_confs(&[0.5])
            .with_model_max_length(32) // CLIP max length, enables auto padding/truncation
            .with_tokenizer_file("sam3/tokenizer.json")
            .with_tokenizer_config_file("sam3/tokenizer_config.json")
            .with_special_tokens_map_file("sam3/special_tokens_map.json")
            .with_config_file("sam3/config.json")
    }

    pub fn sam3_image_predictor() -> Self {
        Self::sam3()
            .with_visual_encoder_file("vision-encoder.onnx")
            .with_textual_encoder_file("text-encoder.onnx")
            .with_encoder_file("geometry-encoder.onnx")
            .with_decoder_file("decoder.onnx")
    }
}
