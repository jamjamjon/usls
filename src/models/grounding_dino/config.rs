/// Model configuration for `GroundingDino`
impl crate::Options {
    pub fn grounding_dino() -> Self {
        Self::default()
            .with_model_name("grounding-dino")
            .with_model_kind(crate::Kind::VisionLanguage)
            .with_model_ixx(0, 0, 1.into()) // TODO: current onnx model does not support bs > 1
            .with_model_ixx(0, 2, 800.into()) // TODO: matters
            .with_model_ixx(0, 3, 1200.into()) // TODO: matters
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_resize_filter("CatmullRom")
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_normalize(true)
            .with_class_confs(&[0.25])
            .with_text_confs(&[0.25])
    }

    pub fn grounding_dino_tiny() -> Self {
        Self::grounding_dino().with_model_file("swint-ogc.onnx")
    }

    pub fn grounding_dino_base() -> Self {
        Self::grounding_dino().with_model_file("swinb-cogcoor.onnx")
    }
}
