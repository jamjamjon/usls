use crate::NAMES_BODY_PARTS_28;

/// Model configuration for `Sapiens`
impl crate::Options {
    pub fn sapiens() -> Self {
        Self::default()
            .with_model_name("sapiens")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 2, 1024.into())
            .with_model_ixx(0, 3, 768.into())
            .with_resize_mode(crate::ResizeMode::FitExact)
            .with_resize_filter("Bilinear")
            .with_image_mean(&[123.5, 116.5, 103.5])
            .with_image_std(&[58.5, 57.0, 57.5])
            .with_normalize(false)
    }

    pub fn sapiens_body_part_segmentation() -> Self {
        Self::sapiens()
            .with_model_task(crate::Task::InstanceSegmentation)
            .with_class_names(&NAMES_BODY_PARTS_28)
    }

    pub fn sapiens_seg_0_3b() -> Self {
        Self::sapiens_body_part_segmentation().with_model_file("seg-0.3b.onnx")
    }

    // pub fn sapiens_seg_0_3b_uint8() -> Self {
    //     Self::sapiens_body_part_segmentation().with_model_file("seg-0.3b-uint8.onnx")
    // }

    // pub fn sapiens_seg_0_3b_fp16() -> Self {
    //     Self::sapiens_body_part_segmentation().with_model_file("seg-0.3b-fp16.onnx")
    // }

    // pub fn sapiens_seg_0_3b_bnb4() -> Self {
    //     Self::sapiens_body_part_segmentation().with_model_file("seg-0.3b-bnb4.onnx")
    // }

    // pub fn sapiens_seg_0_3b_q4f16() -> Self {
    //     Self::sapiens_body_part_segmentation().with_model_file("seg-0.3b-q4f16.onnx")
    // }

    // pub fn sapiens_seg_0_6b_fp16() -> Self {
    //     Self::sapiens_body_part_segmentation().with_model_file("seg-0.6b-fp16.onnx")
    // }
}
