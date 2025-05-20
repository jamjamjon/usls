/// Model configuration for `DINOv2`
impl crate::Config {
    pub fn dinov2() -> Self {
        Self::default()
            .with_name("dinov2")
            .with_model_ixx(0, 0, (1, 1, 8).into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 224.into())
            .with_model_ixx(0, 3, 224.into())
            .with_resize_mode(crate::ResizeMode::FitExact)
            .with_resize_filter("Lanczos3")
            .with_normalize(true)
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_image_mean(&[0.485, 0.456, 0.406])
    }

    pub fn dinov2_small() -> Self {
        Self::dinov2()
            .with_scale(crate::Scale::S)
            .with_model_file("s.onnx")
    }

    pub fn dinov2_base() -> Self {
        Self::dinov2()
            .with_scale(crate::Scale::B)
            .with_model_file("b.onnx")
    }
}
