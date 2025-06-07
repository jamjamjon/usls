use crate::{models::SamKind, Config};

/// Model configuration for `Segment Anything Model`
impl Config {
    /// Creates a base SAM configuration with common settings.
    ///
    /// Sets up default parameters for image preprocessing and model architecture:
    /// - 1024x1024 input resolution
    /// - Adaptive resize mode
    /// - Image normalization parameters
    /// - Contour finding enabled
    pub fn sam() -> Self {
        Self::default()
            .with_name("sam")
            .with_encoder_ixx(0, 0, 1.into())
            .with_encoder_ixx(0, 1, 3.into())
            .with_encoder_ixx(0, 2, 1024.into())
            .with_encoder_ixx(0, 3, 1024.into())
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_resize_filter("Bilinear")
            .with_image_mean(&[123.5, 116.5, 103.5])
            .with_image_std(&[58.5, 57.0, 57.5])
            .with_normalize(false)
            .with_sam_kind(SamKind::Sam)
            .with_sam_low_res_mask(false)
            .with_find_contours(true)
    }

    /// Creates a configuration for SAM v1 base model.
    /// Uses the original ViT-B architecture.  
    pub fn sam_v1_base() -> Self {
        Self::sam()
            .with_encoder_file("sam-vit-b-encoder.onnx")
            .with_decoder_file("sam-vit-b-decoder.onnx")
    }

    // pub fn sam_v1_base_singlemask_decoder() -> Self {
    //     Self::sam().with_decoder_file("sam-vit-b-decoder-singlemask.onnx")
    // }

    /// Creates a configuration for SAM 2.0 tiny model.
    /// Uses a hierarchical architecture with tiny backbone.
    pub fn sam2_tiny() -> Self {
        Self::sam()
            .with_encoder_file("sam2-hiera-tiny-encoder.onnx")
            .with_sam_kind(SamKind::Sam2)
            .with_decoder_file("sam2-hiera-tiny-decoder.onnx")
    }

    /// Creates a configuration for SAM 2.0 small model.
    /// Uses a hierarchical architecture with small backbone.
    pub fn sam2_small() -> Self {
        Self::sam()
            .with_encoder_file("sam2-hiera-small-encoder.onnx")
            .with_decoder_file("sam2-hiera-small-decoder.onnx")
            .with_sam_kind(SamKind::Sam2)
    }

    /// Creates a configuration for SAM 2.0 base-plus model.
    /// Uses a hierarchical architecture with enhanced base backbone.
    pub fn sam2_base_plus() -> Self {
        Self::sam()
            .with_encoder_file("sam2-hiera-base-plus-encoder.onnx")
            .with_decoder_file("sam2-hiera-base-plus-decoder.onnx")
            .with_sam_kind(SamKind::Sam2)
    }

    /// Creates a configuration for MobileSAM tiny model.
    /// Lightweight model optimized for mobile devices.
    pub fn mobile_sam_tiny() -> Self {
        Self::sam()
            .with_encoder_file("mobile-sam-vit-t-encoder.onnx")
            .with_sam_kind(SamKind::MobileSam)
            .with_decoder_file("mobile-sam-vit-t-decoder.onnx")
    }

    /// Creates a configuration for SAM-HQ tiny model.
    /// High-quality variant focused on better mask quality.
    pub fn sam_hq_tiny() -> Self {
        Self::sam()
            .with_encoder_file("sam-hq-vit-t-encoder.onnx")
            .with_sam_kind(SamKind::SamHq)
            .with_decoder_file("sam-hq-vit-t-decoder.onnx")
    }

    /// Creates a configuration for EdgeSAM 3x model.
    /// Edge-based variant optimized for speed and efficiency.
    pub fn edge_sam_3x() -> Self {
        Self::sam()
            .with_encoder_file("edge-sam-3x-encoder.onnx")
            .with_decoder_file("edge-sam-3x-decoder.onnx")
            .with_sam_kind(SamKind::EdgeSam)
    }
}
