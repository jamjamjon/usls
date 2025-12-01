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
            .with_num_dry_run_all(1)
            .with_batch_size_all_min_opt_max(1, 1, 10)
            .with_visual_encoder_ixx(0, 1, 3.into()) // vision-encoder: channels
            .with_visual_encoder_ixx(0, 2, 1008.into()) // vision-encoder: height
            .with_visual_encoder_ixx(0, 3, 1008.into()) // vision-encoder: width
            .with_textual_encoder_ixx(0, 1, 32.into()) // text-encoder: text sequence length
            .with_encoder_ixx(0, 1, (1, 2, 8).into()) // geometry-encoder: input_boxes, num_boxes
            .with_encoder_ixx(1, 1, (1, 2, 8).into()) // geometry-encoder: input_boxes_labels, num_boxes
            .with_decoder_ixx(4, 1, (1, 34, 60).into()) // decoder: prompt_features prompt_len
            .with_decoder_ixx(5, 1, (1, 34, 60).into()) // decoder: prompt_mask prompt_len
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
