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
            .with_visual_encoder_ixx(0, 1, 3) // vision-encoder: channels
            .with_visual_encoder_ixx(0, 2, 1008) // vision-encoder: height
            .with_visual_encoder_ixx(0, 3, 1008) // vision-encoder: width
            .with_textual_encoder_ixx(0, 1, 32) // text-encoder: sequence length
            .with_encoder_ixx(0, 1, (1, 2, 8)) // geometry-encoder: num_boxes
            .with_encoder_ixx(1, 1, (1, 2, 8)) // geometry-encoder: num_boxes
            .with_decoder_ixx(4, 1, (1, 34, 60)) // decoder: prompt_len
            .with_decoder_ixx(5, 1, (1, 34, 60)) // decoder: prompt_len
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
}
