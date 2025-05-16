use crate::{models::SamKind, ModelConfig};

/// Model configuration for `Segment Anything Model`
impl ModelConfig {
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

    pub fn sam_v1_base() -> Self {
        Self::sam()
            .with_encoder_file("sam-vit-b-encoder.onnx")
            .with_decoder_file("sam-vit-b-decoder.onnx")
    }

    // pub fn sam_v1_base_singlemask_decoder() -> Self {
    //     Self::sam().with_decoder_file("sam-vit-b-decoder-singlemask.onnx")
    // }

    pub fn sam2_tiny() -> Self {
        Self::sam()
            .with_encoder_file("sam2-hiera-tiny-encoder.onnx")
            .with_sam_kind(SamKind::Sam2)
            .with_decoder_file("sam2-hiera-tiny-decoder.onnx")
    }

    pub fn sam2_small() -> Self {
        Self::sam()
            .with_encoder_file("sam2-hiera-small-encoder.onnx")
            .with_decoder_file("sam2-hiera-small-decoder.onnx")
            .with_sam_kind(SamKind::Sam2)
    }

    pub fn sam2_base_plus() -> Self {
        Self::sam()
            .with_encoder_file("sam2-hiera-base-plus-encoder.onnx")
            .with_decoder_file("sam2-hiera-base-plus-decoder.onnx")
            .with_sam_kind(SamKind::Sam2)
    }

    pub fn mobile_sam_tiny() -> Self {
        Self::sam()
            .with_encoder_file("mobile-sam-vit-t-encoder.onnx")
            .with_sam_kind(SamKind::MobileSam)
            .with_decoder_file("mobile-sam-vit-t-decoder.onnx")
    }

    pub fn sam_hq_tiny() -> Self {
        Self::sam()
            .with_encoder_file("sam-hq-vit-t-encoder.onnx")
            .with_sam_kind(SamKind::SamHq)
            .with_decoder_file("sam-hq-vit-t-decoder.onnx")
    }

    pub fn edge_sam_3x() -> Self {
        Self::sam()
            .with_encoder_file("edge-sam-3x-encoder.onnx")
            .with_decoder_file("edge-sam-3x-decoder.onnx")
            .with_sam_kind(SamKind::EdgeSam)
    }
}
