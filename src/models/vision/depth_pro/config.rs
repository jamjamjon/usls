/// Model configuration for `DepthPro`
impl crate::Config {
    pub fn depth_pro() -> Self {
        Self::default()
            .with_name("depth-pro")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 1536)
            .with_model_ixx(0, 3, 1536)
            .with_image_mean([0.5, 0.5, 0.5])
            .with_image_std([0.5, 0.5, 0.5])
            .with_normalize(true)
            .with_resize_mode(crate::ResizeMode::FitExact)
            .with_model_file("model.onnx")
    }
}
