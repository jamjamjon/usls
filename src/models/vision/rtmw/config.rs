use crate::NAMES_COCO_KEYPOINTS_133;

///
/// > # RTMW: Real-Time Multi-Person Whole-body Pose Estimation
/// >
/// > Real-time multi-person 2D and 3D whole-body pose estimation with 133 keypoints.
/// >
/// > # Paper & Code
/// >
/// > - **Paper**: [RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation](https://arxiv.org/abs/2407.08634)
/// >
/// > # Model Variants
/// >
/// > - **rtmw-133-m**: Medium model for 133-keypoint whole-body pose estimation
/// > - **rtmw-133-m-384**: Medium model with 384x384 resolution
/// > - **rtmw-133-l**: Large model for 133-keypoint whole-body pose estimation
/// > - **rtmw-133-x**: Extra large model for 133-keypoint whole-body pose estimation
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Whole-body Pose Estimation**: 133-keypoint comprehensive pose detection
/// > - [X] **Multi-person Detection**: Real-time pose estimation for multiple people
/// > - [X] **2D and 3D Support**: Both 2D and 3D pose estimation capabilities
/// > - [X] **High Resolution**: Support for 384x384 input resolution
/// >
/// Model configuration for `RTMW`
///
impl crate::Config {
    /// Base configuration for 133-keypoint whole-body pose estimation
    pub fn rtmpose_133() -> Self {
        Self::rtmpose()
            .with_nk(133)
            .with_keypoint_names(&NAMES_COCO_KEYPOINTS_133)
    }

    /// Medium model for 133-keypoint whole-body pose estimation
    pub fn rtmw_133_m() -> Self {
        Self::rtmpose_133().with_model_file("rtmw-133-m.onnx")
    }

    /// Medium model with 384x384 resolution
    pub fn rtmw_133_m_384() -> Self {
        Self::rtmpose_133().with_model_file("rtmw-133-m-384.onnx")
    }

    /// Large model for 133-keypoint whole-body pose estimation
    pub fn rtmw_133_l() -> Self {
        Self::rtmpose_133().with_model_file("rtmw-133-l.onnx")
    }

    /// Extra large model for 133-keypoint whole-body pose estimation
    pub fn rtmw_133_x() -> Self {
        Self::rtmpose_133().with_model_file("rtmw-133-x.onnx")
    }
}
