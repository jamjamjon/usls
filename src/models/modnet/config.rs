/// Model configuration for `MODNet`
impl crate::Config {
    pub fn modnet() -> Self {
        Self::default()
            .with_name("modnet")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 2, (416, 512, 800).into())
            .with_model_ixx(0, 3, (416, 512, 800).into())
            .with_image_mean(&[0.5, 0.5, 0.5])
            .with_image_std(&[0.5, 0.5, 0.5])
            .with_normalize(true)
    }

    pub fn modnet_photographic() -> Self {
        Self::modnet().with_model_file("photographic-portrait-matting.onnx")
    }
}
