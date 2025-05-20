/// Model configuration for `DepthAnything`
impl crate::Config {
    pub fn depth_anything() -> Self {
        Self::default()
            .with_name("depth-anything")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, (384, 518, 1024).into())
            .with_model_ixx(0, 3, (384, 518, 1024).into())
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_resize_filter("Lanczos3")
            .with_normalize(true)
    }

    pub fn depth_anything_s() -> Self {
        Self::depth_anything().with_scale(crate::Scale::S)
    }

    pub fn depth_anything_v1() -> Self {
        Self::depth_anything().with_version(1.into())
    }

    pub fn depth_anything_v2() -> Self {
        Self::depth_anything().with_version(2.into())
    }

    pub fn depth_anything_v1_small() -> Self {
        Self::depth_anything_v1()
            .with_scale(crate::Scale::S)
            .with_model_file("v1-s.onnx")
    }

    pub fn depth_anything_v2_small() -> Self {
        Self::depth_anything_v2()
            .with_scale(crate::Scale::S)
            .with_model_file("v2-s.onnx")
    }
}
