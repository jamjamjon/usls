///
/// > # DWPose: Effective Whole-body Pose Estimation
/// >
/// > Whole-body pose estimation with two-stage distillation for high accuracy.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose)
/// >
/// > # Model Variants
/// >
/// > - **dwpose-133-t**: Tiny model for real-time pose estimation
/// > - **dwpose-133-s**: Small model for balanced performance
/// > - **dwpose-133-m**: Medium model for high accuracy
/// > - **dwpose-133-l**: Large model for maximum accuracy
/// > - **dwpose-133-l-384**: Large model with 384x384 resolution
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Whole-body Pose Estimation**: 133-keypoint full body pose detection
/// > - [X] **Two-stage Distillation**: High-accuracy pose estimation
/// >
/// Model configuration for `DWPose`
///
impl crate::Config {
    /// Tiny model for real-time pose estimation
    pub fn dwpose_133_t() -> Self {
        Self::rtmpose_133().with_model_file("dwpose-133-t.onnx")
    }

    /// Small model for balanced performance
    pub fn dwpose_133_s() -> Self {
        Self::rtmpose_133().with_model_file("dwpose-133-s.onnx")
    }

    /// Medium model for high accuracy
    pub fn dwpose_133_m() -> Self {
        Self::rtmpose_133().with_model_file("dwpose-133-m.onnx")
    }

    /// Large model for maximum accuracy
    pub fn dwpose_133_l() -> Self {
        Self::rtmpose_133().with_model_file("dwpose-133-l.onnx")
    }

    /// Large model with 384x384 resolution
    pub fn dwpose_133_l_384() -> Self {
        Self::rtmpose_133().with_model_file("dwpose-133-l-384.onnx")
    }
}
