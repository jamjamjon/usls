/// Model configuration for `GroundingDino` Series
impl crate::Config {
    pub fn grounding_dino() -> Self {
        Self::default()
            .with_name("grounding-dino")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 800.into())
            .with_model_ixx(0, 3, 1066.into())
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
        Self::grounding_dino().with_model_file("gdino-tiny.onnx")
    }

    pub fn grounding_dino_base() -> Self {
        Self::grounding_dino().with_model_file("gdino-base.onnx")
    }

    pub fn llmdet_tiny() -> Self {
        Self::grounding_dino().with_model_file("llmdet-tiny.onnx")
    }

    pub fn llmdet_base() -> Self {
        Self::grounding_dino().with_model_file("llmdet-base.onnx")
    }

    pub fn llmdet_large() -> Self {
        Self::grounding_dino().with_model_file("llmdet-large.onnx")
    }

    pub fn mm_gdino_tiny_o365v1_goldg_grit_v3det() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-tiny-o365v1-goldg-grit-v3det.onnx")
    }

    pub fn mm_gdino_tiny_o365v1_goldg_grit() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-tiny-o365v1-goldg-grit.onnx")
    }

    pub fn mm_gdino_tiny_o365v1_goldg_v3det() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-tiny-o365v1-goldg-v3det.onnx")
    }

    pub fn mm_gdino_tiny_o365v1_goldg() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-tiny-o365v1-goldg.onnx")
    }

    pub fn mm_gdino_base_o365v1_goldg_v3det() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-base-o365v1-goldg-v3det.onnx")
    }

    pub fn mm_gdino_base_all() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-base-all.onnx")
    }

    pub fn mm_gdino_large_o365v2_oiv6_goldg() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-large-o365v2-oiv6-goldg.onnx")
    }

    pub fn mm_gdino_large_all() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-large-all.onnx")
    }
}
