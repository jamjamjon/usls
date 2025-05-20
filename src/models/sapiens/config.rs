use crate::NAMES_BODY_PARTS_28;

/// Model configuration for `Sapiens`
impl crate::Config {
    pub fn sapiens() -> Self {
        Self::default()
            .with_name("sapiens")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 2, 1024.into())
            .with_model_ixx(0, 3, 768.into())
            .with_resize_mode(crate::ResizeMode::FitExact)
            .with_image_mean(&[123.5, 116.5, 103.5])
            .with_image_std(&[58.5, 57.0, 57.5])
            .with_normalize(false)
    }

    pub fn sapiens_body_part_segmentation() -> Self {
        Self::sapiens()
            .with_task(crate::Task::InstanceSegmentation)
            .with_class_names(&NAMES_BODY_PARTS_28)
    }

    pub fn sapiens_seg_0_3b() -> Self {
        Self::sapiens_body_part_segmentation().with_model_file("seg-0.3b.onnx")
    }
}
