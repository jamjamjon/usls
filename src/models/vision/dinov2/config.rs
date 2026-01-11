///
/// > # DINOv2: Learning Robust Visual Features without Supervision
/// >
/// > Self-supervised vision transformer for robust visual feature learning.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
/// > - **Paper**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07793)
/// >
/// > # Model Variants
/// >
/// > - **dinov2-small**: Small model for efficient feature extraction
/// > - **dinov2-base**: Base model for balanced performance
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Feature Extraction**: Self-supervised visual feature learning
/// > - [X] **Transfer Learning**: Robust features for downstream tasks
/// >
/// Model configuration for `DINOv2`
///
impl crate::Config {
    /// Base configuration for DINOv2 models
    pub fn dinov2() -> Self {
        Self::default()
            .with_name("dinov2")
            .with_model_ixx(0, 0, (1, 1, 8))
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 224)
            .with_model_ixx(0, 3, 224)
            .with_resize_mode_type(crate::ResizeModeType::FitExact)
            .with_resize_filter(crate::ResizeFilter::Bilinear)
            .with_normalize(true)
            .with_image_std([0.229, 0.224, 0.225])
            .with_image_mean([0.485, 0.456, 0.406])
    }

    /// Small model for efficient feature extraction
    pub fn dinov2_small() -> Self {
        Self::dinov2()
            .with_scale(crate::Scale::S)
            .with_model_file("s.onnx")
    }

    /// Base model for balanced performance
    pub fn dinov2_base() -> Self {
        Self::dinov2()
            .with_scale(crate::Scale::B)
            .with_model_file("b.onnx")
    }
}
