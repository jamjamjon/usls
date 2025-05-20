/// Model configuration for `RMBG`
impl crate::Config {
    pub fn rmbg() -> Self {
        Self::default()
            .with_name("rmbg")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 1024.into())
            .with_model_ixx(0, 3, 1024.into())
    }

    pub fn rmbg1_4() -> Self {
        Self::rmbg()
            .with_image_mean(&[0.5, 0.5, 0.5])
            .with_image_std(&[1., 1., 1.])
            .with_model_file("1.4.onnx")
    }

    pub fn rmbg2_0() -> Self {
        Self::rmbg()
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_model_file("2.0.onnx")
    }
}
