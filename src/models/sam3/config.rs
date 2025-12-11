use crate::{Config, Task};

/// Model configuration for `SAM3-Image` and `SAM3-Tracker`
impl Config {
    /// SAM3 base configuration
    pub fn sam3() -> Self {
        Self::default()
            .with_name("sam3")
            .with_num_dry_run_all(1)
            .with_resize_mode(crate::ResizeMode::FitExact)
            .with_resize_filter("Bilinear")
            .with_image_mean(&[0.5, 0.5, 0.5])
            .with_image_std(&[0.5, 0.5, 0.5])
            .with_normalize(true)
            .with_find_contours(true)
            .with_class_confs(&[0.5])
    }

    /// SAM3 Image Predictor configuration (text + geometry prompts)
    pub fn sam3_image() -> Self {
        Self::sam3()
            .with_task(Task::Sam3Image)
            // Batch sizes: vision=1, text=4, geometry=8, decoder=1
            .with_visual_encoder_batch_min_opt_max(1, 1, 8)
            .with_textual_encoder_batch_min_opt_max(1, 4, 16)
            .with_encoder_batch_min_opt_max(1, 8, 16) // geometry encoder
            .with_decoder_batch_min_opt_max(1, 1, 8) // decoder (memory intensive)
            // Shape configurations
            .with_visual_encoder_ixx(0, 1, 3.into()) // vision-encoder: channels
            .with_visual_encoder_ixx(0, 2, 1008.into()) // vision-encoder: height
            .with_visual_encoder_ixx(0, 3, 1008.into()) // vision-encoder: width
            .with_textual_encoder_ixx(0, 1, 32.into()) // text-encoder: sequence length
            .with_encoder_ixx(0, 1, (1, 2, 8).into()) // geometry-encoder: num_boxes
            .with_encoder_ixx(1, 1, (1, 2, 8).into()) // geometry-encoder: num_boxes
            .with_decoder_ixx(4, 1, (1, 34, 60).into()) // decoder: prompt_len
            .with_decoder_ixx(5, 1, (1, 34, 60).into()) // decoder: prompt_len
            // Tokenizer configs
            .with_model_max_length(32) // CLIP max length, enables auto padding/truncation
            .with_tokenizer_file("sam3/tokenizer.json")
            .with_tokenizer_config_file("sam3/tokenizer_config.json")
            .with_special_tokens_map_file("sam3/special_tokens_map.json")
            .with_config_file("sam3/config.json")
            // Model files
            .with_visual_encoder_file("vision-encoder.onnx")
            .with_textual_encoder_file("text-encoder.onnx")
            .with_encoder_file("geometry-encoder.onnx")
            .with_decoder_file("decoder.onnx")
    }

    /// SAM3 Tracker configuration (point + box prompts)
    ///
    /// ONNX Models:
    /// - vision_encoder.onnx: [B, 3, 1008, 1008] -> embeddings (32x288x288, 64x144x144, 256x72x72)
    /// - prompt_encoder_mask_decoder.onnx: points/boxes + embeddings -> masks
    pub fn sam3_tracker() -> Self {
        Self::sam3()
            .with_task(Task::Sam3Tracker)
            // Batch sizes
            .with_visual_encoder_batch_min_opt_max(1, 1, 4)
            .with_decoder_batch_min_opt_max(1, 1, 4)
            // Vision encoder shape: [B, 3, 1008, 1008]
            .with_visual_encoder_ixx(0, 1, 3.into())
            .with_visual_encoder_ixx(0, 2, 1008.into())
            .with_visual_encoder_ixx(0, 3, 1008.into())
            // Decoder input shapes (dynamic)
            // input_points: [B, 1, num_points, 2]
            .with_decoder_ixx(0, 2, (1, 2, 16).into())
            // input_labels: [B, 1, num_points]
            .with_decoder_ixx(1, 2, (1, 2, 16).into())
            // input_boxes: [B, num_boxes, 4]
            .with_decoder_ixx(2, 1, 1.into())
            // Model files
            .with_visual_encoder_file("tracker-vision-encoder.onnx")
            .with_decoder_file("tracker-prompt-encoder-mask-decoder.onnx")
    }
}
