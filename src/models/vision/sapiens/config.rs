use crate::NAMES_BODY_PARTS_28;

///
/// > # Sapiens: Foundation for Human Vision Models
/// >
/// > Foundation models for human-centric vision tasks developed by Meta AI.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [facebookresearch/sapiens](https://github.com/facebookresearch/sapiens)
/// >
/// > # Model Variants
/// >
/// > - **sapiens-seg-0.3b**: 0.3B parameter body-part segmentation model
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Body-Part Segmentation**: 28-class human body part segmentation
/// > - [ ] **Pose Estimation**: Human keypoint detection
/// > - [ ] **Depth Estimation**: Monocular depth estimation
/// > - [ ] **Surface Normal Estimation**: Surface normal prediction
///
/// Model configuration for `Sapiens`
///
impl crate::Config {
    /// Base configuration for Sapiens models
    pub fn sapiens() -> Self {
        Self::default()
            .with_name("sapiens")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 1024)
            .with_model_ixx(0, 3, 768)
            .with_resize_mode_type(crate::ResizeModeType::FitExact)
            .with_image_mean([123.5, 116.5, 103.5])
            .with_image_std([58.5, 57.0, 57.5])
            .with_normalize(false)
    }

    fn sapiens_body_part_segmentation() -> Self {
        Self::sapiens()
            .with_task(crate::Task::InstanceSegmentation)
            .with_class_names(&NAMES_BODY_PARTS_28)
    }

    /// 0.3B body-part segmentation model
    pub fn sapiens_seg_0_3b() -> Self {
        Self::sapiens_body_part_segmentation().with_model_file("seg-0.3b.onnx")
    }
}
