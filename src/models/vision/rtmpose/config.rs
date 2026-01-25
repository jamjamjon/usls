use crate::{NAMES_COCO_KEYPOINTS_17, NAMES_HALPE_KEYPOINTS_26};

///
/// > # RTMPose: Real-Time Multi-Person Pose Estimation
/// >
/// > Real-time multi-person pose estimation toolkit based on MMPose framework.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose)
/// >
/// > # Model Variants
/// >
/// > - **rtmpose-17-t**: Tiny model for 17-keypoint COCO pose estimation
/// > - **rtmpose-17-s**: Small model for 17-keypoint COCO pose estimation
/// > - **rtmpose-17-m**: Medium model for 17-keypoint COCO pose estimation
/// > - **rtmpose-17-l**: Large model for 17-keypoint COCO pose estimation
/// > - **rtmpose-17-l-384**: Large model with 384x384 resolution
/// > - **rtmpose-17-x**: Extra large model for 17-keypoint COCO pose estimation
/// > - **rtmpose-26-t**: Tiny model for 26-keypoint HALPE pose estimation
/// > - **rtmpose-26-s**: Small model for 26-keypoint HALPE pose estimation
/// > - **rtmpose-26-m**: Medium model for 26-keypoint HALPE pose estimation
/// > - **rtmpose-26-m-384**: Medium model with 384x384 resolution
/// > - **rtmpose-26-l**: Large model for 26-keypoint HALPE pose estimation
/// > - **rtmpose-26-l-384**: Large model with 384x384 resolution
/// > - **rtmpose-26-x**: Extra large model for 26-keypoint HALPE pose estimation
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Multi-Person Pose Estimation**: Real-time pose detection for multiple people
/// > - [X] **COCO Keypoints**: 17-keypoint COCO pose estimation
/// > - [X] **HALPE Keypoints**: 26-keypoint HALPE pose estimation
/// > - [X] **Multiple Resolutions**: Support for 256x192 and 384x384 inputs
/// >
/// Model configuration for `RTMPose`
///
impl crate::Config {
    /// Base configuration for RTMPose models
    pub fn rtmpose() -> Self {
        Self::default()
            .with_name("rtmpose")
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 256)
            .with_model_ixx(0, 3, 192)
            .with_image_mean([123.675, 116.28, 103.53])
            .with_image_std([58.395, 57.12, 57.375])
            .with_normalize(false) // matters!
            .with_keypoint_confs(&[0.35])
    }

    /// Base configuration for 17-keypoint COCO pose estimation
    pub fn rtmpose_17() -> Self {
        Self::rtmpose()
            .with_nk(17)
            .with_keypoint_names(&NAMES_COCO_KEYPOINTS_17)
    }

    /// Tiny model for 17-keypoint COCO pose estimation
    pub fn rtmpose_17_t() -> Self {
        Self::rtmpose_17().with_model_file("rtmpose-17-t.onnx")
    }

    /// Small model for 17-keypoint COCO pose estimation
    pub fn rtmpose_17_s() -> Self {
        Self::rtmpose_17().with_model_file("rtmpose-17-s.onnx")
    }

    /// Medium model for 17-keypoint COCO pose estimation
    pub fn rtmpose_17_m() -> Self {
        Self::rtmpose_17().with_model_file("rtmpose-17-m.onnx")
    }

    /// Large model for 17-keypoint COCO pose estimation
    pub fn rtmpose_17_l() -> Self {
        Self::rtmpose_17().with_model_file("rtmpose-17-l.onnx")
    }

    /// Large model with 384x384 resolution
    pub fn rtmpose_17_l_384() -> Self {
        Self::rtmpose_17().with_model_file("rtmpose-17-l-384.onnx")
    }

    /// Extra large model for 17-keypoint COCO pose estimation
    pub fn rtmpose_17_x() -> Self {
        Self::rtmpose_17().with_model_file("rtmpose-17-x.onnx")
    }

    fn rtmpose_26() -> Self {
        Self::rtmpose()
            .with_nk(26)
            .with_keypoint_names(&NAMES_HALPE_KEYPOINTS_26)
    }

    /// Tiny model for 26-keypoint HALPE pose estimation
    pub fn rtmpose_26_t() -> Self {
        Self::rtmpose_26().with_model_file("rtmpose-26-t.onnx")
    }

    /// Small model for 26-keypoint HALPE pose estimation
    pub fn rtmpose_26_s() -> Self {
        Self::rtmpose_26().with_model_file("rtmpose-26-s.onnx")
    }

    /// Medium model for 26-keypoint HALPE pose estimation
    pub fn rtmpose_26_m() -> Self {
        Self::rtmpose_26().with_model_file("rtmpose-26-m.onnx")
    }

    /// Medium model with 384x384 resolution
    pub fn rtmpose_26_m_384() -> Self {
        Self::rtmpose_26().with_model_file("rtmpose-26-m-384.onnx")
    }

    /// Large model for 26-keypoint HALPE pose estimation
    pub fn rtmpose_26_l() -> Self {
        Self::rtmpose_26().with_model_file("rtmpose-26-l.onnx")
    }

    /// Large model with 384x384 resolution
    pub fn rtmpose_26_l_384() -> Self {
        Self::rtmpose_26().with_model_file("rtmpose-26-l-384.onnx")
    }

    /// Extra large model for 26-keypoint HALPE pose estimation
    pub fn rtmpose_26_x() -> Self {
        Self::rtmpose_26().with_model_file("rtmpose-26-x.onnx")
    }
}
