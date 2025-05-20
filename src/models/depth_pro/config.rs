/// Model configuration for `DepthPro`
impl crate::Config {
    pub fn depth_pro() -> Self {
        Self::default()
            .with_name("depth-pro")
            .with_model_ixx(0, 0, 1.into()) // batch. Note: now only support batch_size = 1
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 1536.into())
            .with_model_ixx(0, 3, 1536.into())
            .with_image_mean(&[0.5, 0.5, 0.5])
            .with_image_std(&[0.5, 0.5, 0.5])
            .with_resize_mode(crate::ResizeMode::FitExact)
            .with_normalize(true)
            .with_model_file("model.onnx")
    }
}
