/// Model configuration for `GroundingDino`
impl crate::Config {
    pub fn grounding_dino() -> Self {
        Self::default()
            .with_name("grounding-dino")
            .with_model_ixx(0, 0, 1.into()) // TODO: current onnx model does not support bs > 1
            .with_model_ixx(0, 2, 800.into()) // TODO: matters
            .with_model_ixx(0, 3, 1200.into()) // TODO: matters
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_resize_filter("CatmullRom")
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_tokenizer_file("grounding-dino/tokenizer.json")
            .with_config_file("grounding-dino/config.json")
            .with_special_tokens_map_file("grounding-dino/special_tokens_map.json")
            .with_tokenizer_config_file("grounding-dino/tokenizer_config.json")
    }

    pub fn grounding_dino_tiny() -> Self {
        Self::grounding_dino().with_model_file("swint-ogc.onnx")
    }

    pub fn grounding_dino_base() -> Self {
        Self::grounding_dino().with_model_file("swinb-cogcoor.onnx")
    }
}
