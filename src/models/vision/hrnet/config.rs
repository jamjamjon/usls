use crate::{NAMES_COCO_KEYPOINTS_133, NAMES_COCO_KEYPOINTS_17};

///
/// > # HRNet: Deep High-Resolution Representation Learning for Human Pose Estimation
/// >
/// > Top-down heatmap-based pose estimator that maintains high-resolution
/// > representations through the whole network.
/// >
/// > # Paper & Code
/// >
/// > - **Paper**: [Deep High-Resolution Representation Learning for Human Pose Estimation](https://arxiv.org/abs/1902.09212)
/// > - **GitHub**: [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/topdown_heatmap)
/// >
/// > # Model Variants
/// >
/// > - **hrnet-w32 / hrnet-w48**: backbone widths (32 / 48 channels)
/// > - **17 keypoints**: COCO body pose estimation
/// > - **133 keypoints**: COCO-WholeBody pose estimation
/// > - **256x192 / 384x288**: supported input resolutions
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Body Pose Estimation**: 17-keypoint COCO pose estimation
/// > - [X] **Whole-body Pose Estimation**: 133-keypoint COCO-WholeBody pose estimation
/// > - [X] **Multiple Backbones**: w32 / w48
/// > - [X] **Multiple Resolutions**: 256x192 and 384x288 inputs
/// >
/// Model configuration for `HRNet`
///
impl crate::Config {
    /// Base configuration for HRNet models (256x192 input)
    pub fn hrnet() -> Self {
        Self::default()
            .with_name("hrnet")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 256)
            .with_model_ixx(0, 3, 192)
            .with_image_mean([123.675, 116.28, 103.53])
            .with_image_std([58.395, 57.12, 57.375])
            .with_normalize(false) // matters!
            .with_keypoint_confs(&[0.35])
    }

    /// HRNet for 384x288 input
    pub fn hrnet_384() -> Self {
        Self::hrnet()
            .with_model_ixx(0, 2, 384)
            .with_model_ixx(0, 3, 288)
    }

    /// Base configuration for 17-keypoint COCO body pose estimation
    pub fn hrnet_17() -> Self {
        Self::hrnet()
            .with_nk(17)
            .with_keypoint_names(&NAMES_COCO_KEYPOINTS_17)
    }

    /// Base configuration for 133-keypoint COCO-WholeBody pose estimation
    pub fn hrnet_133() -> Self {
        Self::hrnet_384()
            .with_nk(133)
            .with_keypoint_names(&NAMES_COCO_KEYPOINTS_133)
    }

    /// HRNet-w32, 17-keypoint COCO body, 256x192
    pub fn hrnet_w32_17() -> Self {
        Self::hrnet_17().with_model_file("hrnet-w32-coco-256x192.onnx")
    }

    /// HRNet-w32, 17-keypoint COCO body, 384x288
    pub fn hrnet_w32_17_384() -> Self {
        Self::hrnet_17()
            .with_model_ixx(0, 2, 384)
            .with_model_ixx(0, 3, 288)
            .with_model_file("hrnet-w32-coco-384x288.onnx")
    }

    /// HRNet-w48, 17-keypoint COCO body, 256x192
    pub fn hrnet_w48_17() -> Self {
        Self::hrnet_17().with_model_file("hrnet-w48-coco-256x192.onnx")
    }

    /// HRNet-w48, 17-keypoint COCO body, 384x288
    pub fn hrnet_w48_17_384() -> Self {
        Self::hrnet_17()
            .with_model_ixx(0, 2, 384)
            .with_model_ixx(0, 3, 288)
            .with_model_file("hrnet-w48-coco-384x288.onnx")
    }

    /// HRNet-w32, 133-keypoint COCO-WholeBody, 384x288
    pub fn hrnet_w32_133() -> Self {
        Self::hrnet_133().with_model_file("hrnet-w32-coco-wholebody-384x288.onnx")
    }

    /// HRNet-w48, 133-keypoint COCO-WholeBody, 384x288
    pub fn hrnet_w48_133() -> Self {
        Self::hrnet_133().with_model_file("hrnet-w48-coco-wholebody-384x288.onnx")
    }
}
