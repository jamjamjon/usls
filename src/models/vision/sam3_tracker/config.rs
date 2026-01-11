use crate::{Config, Task};

///
/// > # SAM3: Segment Anything with Concepts
/// >
/// > Multimodal segmentation model supporting text, bounding box, and combined prompts with tracking.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
/// > - **ONNX Models**:[GitHub Release](https://github.com/jamjamjon/assets/releases/tag/sam3)
/// >
/// > # Model Variants
/// >
/// > - **sam3-tracker**: Multi-object tracking with point and box prompts
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Multi-object Tracking**: Track multiple objects across frames
/// > - [X] **Multimodal Prompts**: Support for point, box, and text prompts
/// > - [X] **High Resolution**: 1008x1008 input resolution
/// > - [X] **Batch Processing**: Support for batch inference
/// >
/// Model configuration for `SAM3-Image` and `SAM3-Tracker`
///
impl Config {
    /// Base configuration for SAM3 models
    pub fn sam3() -> Self {
        Self::default()
            .with_name("sam3")
            .with_num_dry_run_all(3)
            .with_resize_mode_type(crate::ResizeModeType::FitExact)
            .with_resize_filter(crate::ResizeFilter::Bilinear)
            .with_image_mean([0.5, 0.5, 0.5])
            .with_image_std([0.5, 0.5, 0.5])
            .with_normalize(true)
            .with_find_contours(false)
            .with_class_confs(&[0.5])
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
            .with_visual_encoder_ixx(0, 1, 3)
            .with_visual_encoder_ixx(0, 2, 1008)
            .with_visual_encoder_ixx(0, 3, 1008)
            // Decoder input shapes (dynamic)
            // input_points: [B, 1, num_points, 2]
            .with_decoder_ixx(0, 2, (1, 2, 16))
            // input_labels: [B, 1, num_points]
            .with_decoder_ixx(1, 2, (1, 2, 16))
            // input_boxes: [B, num_boxes, 4]
            .with_decoder_ixx(2, 1, 1)
            // Model files
            .with_visual_encoder_file("tracker-vision-encoder.onnx")
            .with_decoder_file("tracker-prompt-encoder-mask-decoder.onnx")
    }
}
