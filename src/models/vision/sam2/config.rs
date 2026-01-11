use crate::Config;

///
/// > # SAM 2.1: Advanced Segment Anything Model
/// >
/// > Enhanced version of Segment Anything Model with improved performance and capabilities.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
/// > - **Paper**: [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.06754)
/// >
/// > # Model Variants
/// >
/// > - **sam2-1-tiny**: Tiny model optimized for speed
/// > - **sam2-1-small**: Small model with balanced performance
/// > - **sam2-1-base-plus**: Enhanced base model with improved quality
/// > - **sam2-1-large**: Large model with highest segmentation quality
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Video Segmentation**: Segment objects across video frames
/// > - [X] **Improved Performance**: Enhanced segmentation quality over SAM 2.0
/// > - [X] **Hierarchical Architecture**: Advanced backbone design
/// > - [X] **Multiple Scales**: Various model sizes for different use cases
/// >
/// Model configuration for `SAM2.1`
///
impl Config {
    /// SAM 2.1 tiny model optimized for speed
    pub fn sam2_1_tiny() -> Self {
        Self::sam()
            .with_encoder_file("sam2.1-hiera-tiny-encoder.onnx")
            .with_decoder_file("sam2.1-hiera-tiny-decoder.onnx")
    }

    /// SAM 2.1 small model with balanced performance
    pub fn sam2_1_small() -> Self {
        Self::sam()
            .with_encoder_file("sam2.1-hiera-small-encoder.onnx")
            .with_decoder_file("sam2.1-hiera-small-decoder.onnx")
    }

    /// SAM 2.1 base-plus model with improved quality
    pub fn sam2_1_base_plus() -> Self {
        Self::sam()
            .with_encoder_file("sam2.1-hiera-base-plus-encoder.onnx")
            .with_decoder_file("sam2.1-hiera-base-plus-decoder.onnx")
    }

    /// SAM 2.1 large model with highest segmentation quality
    pub fn sam2_1_large() -> Self {
        Self::sam()
            .with_encoder_file("sam2.1-hiera-large-encoder.onnx")
            .with_decoder_file("sam2.1-hiera-large-decoder.onnx")
    }
}
