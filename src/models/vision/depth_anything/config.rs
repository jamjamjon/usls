///
/// > # Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data
/// >
/// > Foundation model for monocular depth estimation using large-scale unlabeled data.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub v1**: [LiheYoung/Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
/// > - **GitHub v2**: [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
/// > - **Paper**: [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2301.12041)
/// >
/// > # Model Variants
/// >
/// > - **depth-anything-v1-small**: V1 small model for monocular depth
/// > - **depth-anything-v1-base**: V1 base model for monocular depth
/// > - **depth-anything-v1-large**: V1 large model for monocular depth
/// > - **depth-anything-v2-small**: V2 small model for monocular depth
/// > - **depth-anything-v2-base**: V2 base model for monocular depth
/// > - **depth-anything-v2-large**: V2 large model for monocular depth
/// > - **depth-anything-v3-mono-large**: V3 large monocular depth model
/// > - **depth-anything-v3-metric-large**: V3 large metric depth model
/// > - **depth-anything-v3-small**: V3 small multiocular depth model
/// > - **depth-anything-v3-base**: V3 base multiocular depth model
/// > - **depth-anything-v3-large**: V3 large multiocular depth model
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Monocular Depth Estimation**: Single-image depth prediction
/// > - [X] **Metric Depth Estimation**: Scale-aware depth prediction
/// > - [X] **Multiocular Depth Estimation**: Multi-image depth prediction
/// >
/// Model configuration for `DepthAnything`
///
impl crate::Config {
    /// Base configuration for Depth Anything models
    pub fn depth_anything() -> Self {
        Self::default()
            .with_name("depth-anything")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, (378, 518, 1008))
            .with_model_ixx(0, 3, (378, 518, 1008))
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
            .with_resize_filter(crate::ResizeFilter::Bilinear)
            .with_normalize(true)
            .with_task(crate::Task::MonocularDepthEstimation)
    }

    /// V1 small model for monocular depth
    pub fn depth_anything_v1_small() -> Self {
        Self::depth_anything()
            .with_version(1.into())
            .with_scale(crate::Scale::S)
            .with_model_file("v1-small.onnx")
    }

    /// V1 base model for monocular depth
    pub fn depth_anything_v1_base() -> Self {
        Self::depth_anything()
            .with_version(1.into())
            .with_scale(crate::Scale::B)
            .with_model_file("v1-base.onnx")
    }

    /// V1 large model for monocular depth
    pub fn depth_anything_v1_large() -> Self {
        Self::depth_anything()
            .with_version(1.into())
            .with_scale(crate::Scale::L)
            .with_model_file("v1-large.onnx")
    }

    /// V2 small model for monocular depth
    pub fn depth_anything_v2_small() -> Self {
        Self::depth_anything()
            .with_version(2.into())
            .with_scale(crate::Scale::S)
            .with_model_file("v2-small.onnx")
    }

    /// V2 base model for monocular depth
    pub fn depth_anything_v2_base() -> Self {
        Self::depth_anything()
            .with_version(2.into())
            .with_scale(crate::Scale::B)
            .with_model_file("v2-base.onnx")
    }

    /// V2 large model for monocular depth
    pub fn depth_anything_v2_large() -> Self {
        Self::depth_anything()
            .with_version(2.into())
            .with_scale(crate::Scale::L)
            .with_model_file("v2-large.onnx")
    }

    /// V3 large monocular depth model
    pub fn depth_anything_v3_mono_large() -> Self {
        Self::depth_anything()
            .with_version(3.into())
            .with_scale(crate::Scale::L)
            .with_model_file("DA3MONO-LARGE.onnx")
    }

    /// V3 large metric depth model
    pub fn depth_anything_v3_metric_large() -> Self {
        Self::depth_anything()
            .with_version(3.into())
            .with_scale(crate::Scale::L)
            .with_task(crate::Task::MetricDepthEstimation)
            .with_model_file("DA3METRIC-LARGE.onnx")
    }

    fn depth_anything_v3_multi() -> Self {
        Self::depth_anything()
            .with_version(3.into())
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 1)
            .with_model_ixx(0, 2, 3)
            .with_model_ixx(0, 3, (378, 518, 1008))
            .with_model_ixx(0, 4, (378, 518, 1008))
            .with_task(crate::Task::MultiocularDepthEstimation)
    }

    /// V3 small multiocular depth model
    pub fn depth_anything_v3_small() -> Self {
        Self::depth_anything_v3_multi()
            .with_scale(crate::Scale::S)
            .with_model_file("DA3-SMALL.onnx")
    }

    /// V3 base multiocular depth model
    pub fn depth_anything_v3_base() -> Self {
        Self::depth_anything_v3_multi()
            .with_scale(crate::Scale::B)
            .with_model_file("DA3-BASE.onnx")
    }

    /// V3 large multiocular depth model
    pub fn depth_anything_v3_large() -> Self {
        Self::depth_anything_v3_multi()
            .with_scale(crate::Scale::L)
            .with_model_file("DA3-LARGE-1.1.onnx")
    }
}
