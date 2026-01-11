use crate::NAMES_COCO_KEYPOINTS_17;

///
/// > # RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation
/// >
/// > High-performance one-stage real-time multi-person pose estimation with optimized inference.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [open-mmlab/mmpose/rtmo](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo)
/// >
/// > # Model Variants
/// >
/// > - **rtmo-t**: Tiny model with 416x416 input resolution
/// > - **rtmo-s**: Small model with 640x640 input resolution
/// > - **rtmo-m**: Medium model with 640x640 input resolution
/// > - **rtmo-l**: Large model with 640x640 input resolution
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Multi-Person Pose Estimation**: Real-time multi-person pose detection
/// > - [X] **One-Stage Architecture**: Efficient single-stage pose estimation
/// > - [X] **17 Keypoints**: COCO format keypoint detection
/// > - [X] **Multi-Scale Support**: Various model sizes for different performance needs
/// >
/// Model configuration for `RTMO`
///
impl crate::Config {
    /// Base configuration for RTMO models
    pub fn rtmo() -> Self {
        Self::default()
            .with_name("rtmo")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 640)
            .with_model_ixx(0, 3, 640)
            .with_resize_mode_type(crate::ResizeModeType::FitAdaptive)
            .with_resize_filter(crate::ResizeFilter::CatmullRom)
            .with_normalize(false)
            .with_nk(17)
            .with_class_confs(&[0.3])
            .with_keypoint_confs(&[0.5])
            .with_keypoint_names(&NAMES_COCO_KEYPOINTS_17)
    }

    /// Tiny model with 416x416 input resolution
    pub fn rtmo_t() -> Self {
        Self::rtmo()
            .with_model_ixx(0, 2, 416)
            .with_model_ixx(0, 3, 416)
            .with_model_file("t.onnx")
    }

    /// Small model with 640x640 input resolution
    pub fn rtmo_s() -> Self {
        Self::rtmo().with_model_file("s.onnx")
    }

    /// Medium model with 640x640 input resolution
    pub fn rtmo_m() -> Self {
        Self::rtmo().with_model_file("m.onnx")
    }

    /// Large model with 640x640 input resolution
    pub fn rtmo_l() -> Self {
        Self::rtmo().with_model_file("l.onnx")
    }
}
