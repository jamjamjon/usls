/// Model configuration for `BEiT`
impl crate::Config {
    pub fn beit() -> Self {
        Self::default()
            .with_name("beit")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 224.into())
            .with_model_ixx(0, 3, 224.into())
            .with_image_mean(&[0.5, 0.5, 0.5])
            .with_image_std(&[0.5, 0.5, 0.5])
            .with_normalize(true)
            .with_apply_softmax(true)
            .with_class_names(&crate::NAMES_IMAGENET_1K)
    }

    pub fn beit_base() -> Self {
        Self::beit().with_model_file("b.onnx")
    }

    pub fn beit_large() -> Self {
        Self::beit().with_model_file("l.onnx")
    }
}
