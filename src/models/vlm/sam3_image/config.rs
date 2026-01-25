use crate::{Config, Task};

///
/// > # SAM3: Segment Anything with Concepts
/// >
/// > Multimodal segmentation model supporting text, bounding box, and combined prompts for advanced image understanding.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
/// > - **ONNX Models**: [GitHub Release](https://github.com/jamjamjon/assets/releases/tag/sam3)
/// >
/// > # Model Variants
/// >
/// > - **sam3-image**: Multimodal image segmentation with text and geometry prompts
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Multimodal Prompts**: Support for text, bounding box, and combined prompts
/// > - [X] **High-Resolution Processing**: 1008x1008 image processing
/// > - [X] **Multi-Encoder Architecture**: Separate vision, text, and geometry encoders
/// > - [X] **Flexible Batching**: Dynamic batch sizes for different components
/// >
/// Model configuration for `SAM3-Image` and `SAM3-Tracker`
///
impl Config {
    /// SAM3 Image Predictor configuration (text + box prompts)
    pub fn sam3_image() -> Self {
        Self::sam3()
            .with_task(Task::Sam3Image)
            .with_resize_mode_type(crate::ResizeModeType::FitExact)
            // ---- Tokenizer configs ----
            .with_tokenizer_file("sam3/tokenizer.json")
            .with_tokenizer_config_file("sam3/tokenizer_config.json")
            .with_special_tokens_map_file("sam3/special_tokens_map.json")
            .with_config_file("sam3/config.json")
            .with_model_max_length(32)
            // ---- Model files ----
            .with_visual_encoder_file("vision-encoder.onnx")
            .with_textual_encoder_file("text-encoder.onnx")
            .with_decoder_file("geo-encoder-mask-decoder.onnx")
            // ---- vision encoder: [batch, 3, 1008, 1008] ----
            .with_visual_encoder_batch_min_opt_max(1, 1, 4)
            // ---- text encoder: input_ids: [batch, 32], attention_mask: [batch, 32] ----
            .with_textual_encoder_batch_min_opt_max(1, 1, 8)
            // ---- decoder (with integrated geometry encoder) ----
            // inputs:
            // - fpn_feat_0:[batch, 256, 288, 288],
            // - fpn_feat_1:[batch, 256, 144, 144],
            // - fpn_feat_2:[batch, 256, 72, 72],
            // - fpn_pos_2:[batch, 256, 72, 72],
            // - text_features:[batch, 32, 256],
            // - text_mask:[batch, 32],
            // - input_boxes:[batch, num_boxes, 4],
            // - input_boxes_labels:[batch, num_boxes]
            .with_decoder_batch_min_opt_max(1, 1, 4)
            .with_decoder_ixx(6, 1, (1, 1, 8)) // input_boxes
            .with_decoder_ixx(7, 1, (1, 1, 8)) // input_boxes_labels
    }
}
