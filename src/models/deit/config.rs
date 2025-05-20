use crate::NAMES_IMAGENET_1K;

/// Model configuration for `DeiT`
impl crate::Config {
    pub fn deit() -> Self {
        Self::default()
            .with_name("deit")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 224.into())
            .with_model_ixx(0, 3, 224.into())
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_normalize(true)
            .with_apply_softmax(true)
            .with_class_names(&NAMES_IMAGENET_1K)
    }

    pub fn deit_tiny_distill() -> Self {
        Self::deit().with_model_file("t-distill.onnx")
    }

    pub fn deit_small_distill() -> Self {
        Self::deit().with_model_file("s-distill.onnx")
    }

    pub fn deit_base_distill() -> Self {
        Self::deit().with_model_file("b-distill.onnx")
    }
}
