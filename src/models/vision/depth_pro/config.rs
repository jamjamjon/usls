///
/// > # Depth Pro: Sharp Monocular Metric Depth
/// >
/// > High-quality monocular metric depth estimation in under a second.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro)
/// > - **Paper**: [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073)
/// >
/// > # Model Variants
/// >
/// > - **depth-pro**: Apple's high-quality depth estimation model
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Metric Depth Estimation**: Scale-aware high-quality depth prediction
/// > - [X] **Fast Inference**: Under one second processing time
/// >
/// Model configuration for `DepthPro`
///
impl crate::Config {
    /// Apple's high-quality depth estimation model
    pub fn depth_pro() -> Self {
        Self::default()
            .with_name("depth-pro")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 1536)
            .with_model_ixx(0, 3, 1536)
            .with_image_mean([0.5, 0.5, 0.5])
            .with_image_std([0.5, 0.5, 0.5])
            .with_normalize(true)
            .with_resize_mode_type(crate::ResizeModeType::FitExact)
            .with_model_file("model.onnx")
    }
}
