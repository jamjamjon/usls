use crate::{ResizeAlg, ResizeFilter, Task, NAMES_COCO_KEYPOINTS_17};

const ECPOSE_RELEASE: &str = "https://github.com/wep21/assets/releases/download/ecpose";

///
/// > # ECPose: EdgeCrafter Multi-Person Pose Estimation
/// >
/// > EdgeCrafter pose models exported to ONNX with DETR-style fixed-query outputs.
/// >
/// > # Model Variants
/// >
/// > - **ecpose-s**: Small multi-person COCO-17 pose model
/// > - **ecpose-m**: Medium multi-person COCO-17 pose model
/// > - **ecpose-l**: Large multi-person COCO-17 pose model
/// > - **ecpose-x**: Extra large multi-person COCO-17 pose model
/// >
/// > # Precision / File naming
/// >
/// > Assets are hosted on the `wep21/assets` GitHub releases (tag `ecpose`).
/// > FP32 weights use `ecpose-{scale}.onnx`; non-FP32 weights follow the standard
/// > automatic suffix resolution via [`crate::Config::with_model_dtype`].
///
/// Model configuration for `ECPose`
///
impl crate::Config {
    /// Base configuration for ECPose models.
    pub fn ecpose() -> Self {
        Self::default()
            .with_name("ecpose")
            .with_task(Task::KeypointsDetection)
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 640)
            .with_model_ixx(0, 3, 640)
            .with_model_ixx(1, 1, 2)
            .with_resize_mode_type(crate::ResizeModeType::FitExact)
            .with_resize_alg(ResizeAlg::Interpolation(ResizeFilter::Bilinear))
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
            .with_class_confs(&[0.5])
            .with_nk(17)
            .with_keypoint_names(&NAMES_COCO_KEYPOINTS_17)
    }

    /// Small multi-person COCO-17 pose model.
    pub fn ecpose_s() -> Self {
        Self::ecpose().with_model_file(format!("{ECPOSE_RELEASE}/ecpose-s.onnx"))
    }

    /// Medium multi-person COCO-17 pose model.
    pub fn ecpose_m() -> Self {
        Self::ecpose().with_model_file(format!("{ECPOSE_RELEASE}/ecpose-m.onnx"))
    }

    /// Large multi-person COCO-17 pose model.
    pub fn ecpose_l() -> Self {
        Self::ecpose().with_model_file(format!("{ECPOSE_RELEASE}/ecpose-l.onnx"))
    }

    /// Extra large multi-person COCO-17 pose model.
    pub fn ecpose_x() -> Self {
        Self::ecpose().with_model_file(format!("{ECPOSE_RELEASE}/ecpose-x.onnx"))
    }
}
