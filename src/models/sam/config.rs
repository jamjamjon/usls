use crate::{models::SamKind, Options};

/// Model configuration for `Segment Anything Model`
impl Options {
    pub fn sam() -> Self {
        Self::default()
            .with_model_name("sam")
            .with_model_ixx(0, 0, 1.into())
    }

    pub fn sam_encoder() -> Self {
        Self::sam()
            .with_model_ixx(0, 2, 1024.into())
            .with_model_ixx(0, 3, 1024.into())
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_resize_filter("Bilinear")
            .with_image_mean(&[123.5, 116.5, 103.5])
            .with_image_std(&[58.5, 57.0, 57.5])
            .with_normalize(false)
            .with_sam_kind(SamKind::Sam)
            .with_low_res_mask(false)
            .with_find_contours(true)
    }

    pub fn sam_decoder() -> Self {
        Self::sam()
    }

    pub fn sam_v1_base_encoder() -> Self {
        Self::sam_encoder().with_model_file("sam-vit-b-encoder.onnx")
    }

    pub fn sam_v1_base_decoder() -> Self {
        Self::sam_decoder().with_model_file("sam-vit-b-decoder.onnx")
    }

    pub fn sam_v1_base_singlemask_decoder() -> Self {
        Self::sam_decoder().with_model_file("sam-vit-b-decoder-singlemask.onnx")
    }

    pub fn sam2_tiny_encoder() -> Self {
        Self::sam_encoder()
            .with_model_file("sam2-hiera-tiny-encoder.onnx")
            .with_sam_kind(SamKind::Sam2)
    }

    pub fn sam2_tiny_decoder() -> Self {
        Self::sam_decoder().with_model_file("sam2-hiera-tiny-decoder.onnx")
    }

    pub fn sam2_small_encoder() -> Self {
        Self::sam_encoder()
            .with_model_file("sam2-hiera-small-encoder.onnx")
            .with_sam_kind(SamKind::Sam2)
    }

    pub fn sam2_small_decoder() -> Self {
        Self::sam_decoder().with_model_file("sam2-hiera-small-decoder.onnx")
    }

    pub fn sam2_base_plus_encoder() -> Self {
        Self::sam_encoder()
            .with_model_file("sam2-hiera-base-plus-encoder.onnx")
            .with_sam_kind(SamKind::Sam2)
    }

    pub fn sam2_base_plus_decoder() -> Self {
        Self::sam_decoder().with_model_file("sam2-hiera-base-plus-decoder.onnx")
    }

    pub fn mobile_sam_tiny_encoder() -> Self {
        Self::sam_encoder()
            .with_model_file("mobile-sam-vit-t-encoder.onnx")
            .with_sam_kind(SamKind::MobileSam)
    }

    pub fn mobile_sam_tiny_decoder() -> Self {
        Self::sam_decoder().with_model_file("mobile-sam-vit-t-decoder.onnx")
    }

    pub fn sam_hq_tiny_encoder() -> Self {
        Self::sam_encoder()
            .with_model_file("sam-hq-vit-t-encoder.onnx")
            .with_sam_kind(SamKind::SamHq)
    }

    pub fn sam_hq_tiny_decoder() -> Self {
        Self::sam_decoder().with_model_file("sam-hq-vit-t-decoder.onnx")
    }

    pub fn edge_sam_3x_encoder() -> Self {
        Self::sam_encoder()
            .with_model_file("edge-sam-3x-encoder.onnx")
            .with_sam_kind(SamKind::EdgeSam)
    }

    pub fn edge_sam_3x_decoder() -> Self {
        Self::sam_decoder().with_model_file("edge-sam-3x-decoder.onnx")
    }
}
