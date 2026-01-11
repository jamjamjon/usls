use crate::{Config, SamKind};

///
/// > # Segment Anything Model
/// >
/// > Foundation model for promptable image segmentation with zero-shot generalization.
/// >
/// > # Paper & Code
/// >
/// > - **SAM v1**: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
/// > - **SAM v2**: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
/// > - **MobileSAM**: [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
/// > - **EdgeSAM**: [chongzhou96/EdgeSAM](https://github.com/chongzhou96/EdgeSAM)
/// > - **SAM-HQ**: [SysCV/sam-hq](https://github.com/SysCV/sam-hq)
/// > - **Paper**: [Segment Anything](https://arxiv.org/abs/2304.02643)
/// >
/// > # Model Variants
/// >
/// > - **sam-v1-base**: Original SAM v1 base model with ViT-B architecture
/// > - **sam2-tiny**: SAM 2.0 tiny model with hierarchical architecture
/// > - **sam2-small**: SAM 2.0 small model with hierarchical architecture
/// > - **sam2-base-plus**: SAM 2.0 base-plus model with enhanced backbone
/// > - **mobile-sam-tiny**: MobileSAM tiny model optimized for mobile devices
/// > - **sam-hq-tiny**: SAM-HQ tiny model focused on high-quality masks
/// > - **edge-sam-3x**: EdgeSAM 3x model optimized for speed and efficiency
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Promptable Segmentation**: Point, box, and text prompts
/// > - [X] **Zero-shot Generalization**: Segment any object without training
/// > - [X] **Mobile Optimization**: Lightweight variants for edge devices
/// > - [X] **High-quality Masks**: HQ variants for better mask quality
/// >
/// Model configuration for `Segment Anything Model`
///
impl Config {
    /// Base configuration for SAM models with common settings
    pub fn sam() -> Self {
        Self::default()
            .with_name("sam")
            .with_encoder_ixx(0, 0, 1)
            .with_encoder_ixx(0, 1, 3)
            .with_encoder_ixx(0, 2, 1024)
            .with_encoder_ixx(0, 3, 1024)
            .with_resize_mode_type(crate::ResizeModeType::FitAdaptive)
            .with_resize_filter(crate::ResizeFilter::Bilinear)
            .with_image_mean([123.5, 116.5, 103.5])
            .with_image_std([58.5, 57.0, 57.5])
            .with_normalize(false)
            .with_sam_kind(SamKind::Sam)
            .with_sam_low_res_mask(false)
            .with_find_contours(false)
    }

    /// Original SAM v1 base model with ViT-B architecture
    pub fn sam_v1_base() -> Self {
        Self::sam()
            .with_encoder_file("sam-vit-b-encoder.onnx")
            .with_decoder_file("sam-vit-b-decoder.onnx")
    }

    /// SAM 2.0 tiny model with hierarchical architecture
    pub fn sam2_tiny() -> Self {
        Self::sam()
            .with_sam_kind(SamKind::Sam2)
            .with_encoder_file("sam2-hiera-tiny-encoder.onnx")
            .with_decoder_file("sam2-hiera-tiny-decoder.onnx")
    }

    /// SAM 2.0 small model with hierarchical architecture
    pub fn sam2_small() -> Self {
        Self::sam()
            .with_sam_kind(SamKind::Sam2)
            .with_encoder_file("sam2-hiera-small-encoder.onnx")
            .with_decoder_file("sam2-hiera-small-decoder.onnx")
    }

    /// SAM 2.0 base-plus model with enhanced backbone
    pub fn sam2_base_plus() -> Self {
        Self::sam()
            .with_sam_kind(SamKind::Sam2)
            .with_encoder_file("sam2-hiera-base-plus-encoder.onnx")
            .with_decoder_file("sam2-hiera-base-plus-decoder.onnx")
    }

    /// MobileSAM tiny model optimized for mobile devices
    pub fn mobile_sam_tiny() -> Self {
        Self::sam()
            .with_sam_kind(SamKind::MobileSam)
            .with_encoder_file("mobile-sam-vit-t-encoder.onnx")
            .with_decoder_file("mobile-sam-vit-t-decoder.onnx")
    }

    /// SAM-HQ tiny model focused on high-quality masks
    pub fn sam_hq_tiny() -> Self {
        Self::sam()
            .with_sam_kind(SamKind::SamHq)
            .with_encoder_file("sam-hq-vit-t-encoder.onnx")
            .with_decoder_file("sam-hq-vit-t-decoder.onnx")
    }

    /// EdgeSAM 3x model optimized for speed and efficiency
    pub fn edge_sam_3x() -> Self {
        Self::sam()
            .with_encoder_file("edge-sam-3x-encoder.onnx")
            .with_decoder_file("edge-sam-3x-decoder.onnx")
            .with_sam_kind(SamKind::EdgeSam)
    }
}
