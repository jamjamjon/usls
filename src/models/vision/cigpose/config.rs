use crate::{NAMES_COCO_KEYPOINTS_133, NAMES_COCO_KEYPOINTS_17};

const CIGPOSE_RELEASE: &str = "https://github.com/wep21/assets/releases/download/cigpose";

///
/// > # CIGPose
/// >
/// > SimCC-based human pose estimation models exported to ONNX.
/// >
/// > # Model Variants
/// >
/// > - **cigpose-17-l**: Large model for 17-keypoint COCO pose estimation
/// > - **cigpose-17-l-384**: Large model for 17-keypoint COCO pose estimation with 384x288 input
/// > - **cigpose-133-l**: Large model for 133-keypoint COCO-WholeBody pose estimation
/// > - **cigpose-133-l-384**: Large model for 133-keypoint COCO-WholeBody pose estimation with 384x288 input
/// > - **cigpose-133-x-384**: Extra large model for 133-keypoint COCO-WholeBody pose estimation with 384x288 input
/// >
/// Model configuration for `CIGPose`
///
impl crate::Config {
    /// Base configuration for CIGPose models.
    pub fn cigpose() -> Self {
        Self::rtmpose().with_name("cigpose")
    }

    /// Base configuration for 17-keypoint COCO pose estimation.
    pub fn cigpose_17() -> Self {
        Self::cigpose()
            .with_nk(17)
            .with_keypoint_names(&NAMES_COCO_KEYPOINTS_17)
    }

    /// Base configuration for 133-keypoint COCO-WholeBody pose estimation.
    pub fn cigpose_133() -> Self {
        Self::cigpose()
            .with_nk(133)
            .with_keypoint_names(&NAMES_COCO_KEYPOINTS_133)
    }

    /// Large model for 17-keypoint COCO pose estimation.
    pub fn cigpose_17_l() -> Self {
        Self::cigpose_17().with_model_file(format!("{CIGPOSE_RELEASE}/cigpose-17-l.onnx"))
    }

    /// Large model for 17-keypoint COCO pose estimation with 384x288 input.
    pub fn cigpose_17_l_384() -> Self {
        Self::cigpose_17()
            .with_model_ixx(0, 2, 384)
            .with_model_ixx(0, 3, 288)
            .with_model_file(format!("{CIGPOSE_RELEASE}/cigpose-17-l-384.onnx"))
    }

    /// Large model for 133-keypoint COCO-WholeBody pose estimation.
    pub fn cigpose_133_l() -> Self {
        Self::cigpose_133().with_model_file(format!("{CIGPOSE_RELEASE}/cigpose-133-l.onnx"))
    }

    /// Large model for 133-keypoint COCO-WholeBody pose estimation with 384x288 input.
    pub fn cigpose_133_l_384() -> Self {
        Self::cigpose_133()
            .with_model_ixx(0, 2, 384)
            .with_model_ixx(0, 3, 288)
            .with_model_file(format!("{CIGPOSE_RELEASE}/cigpose-133-l-384.onnx"))
    }

    /// Extra large model for 133-keypoint COCO-WholeBody pose estimation with 384x288 input.
    pub fn cigpose_133_x_384() -> Self {
        Self::cigpose_133()
            .with_model_ixx(0, 2, 384)
            .with_model_ixx(0, 3, 288)
            .with_model_file(format!("{CIGPOSE_RELEASE}/cigpose-133-x-384.onnx"))
    }
}
