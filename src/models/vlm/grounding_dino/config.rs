///
/// > # Grounding DINO Series: Open-Set Detection with Language Guidance
/// >
/// > Open-set object detection models that use language descriptions to guide detection without predefined classes.
/// >
/// > # Paper & Code
/// >
/// > - **GroundingDINO**: [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | [Paper](https://arxiv.org/abs/2303.05499)
/// > - **mm-GroundingDINO**: [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md) | [Paper](https://arxiv.org/abs/2401.02361)
/// > - **LLMDet**: [iSEE-Laboratory/LLMDet](https://github.com/iSEE-Laboratory/LLMDet) | [Paper](https://arxiv.org/abs/2501.18954)
/// >
/// > # Model Variants
/// >
/// > - **grounding-dino-tiny**: Tiny GroundingDINO model
/// > - **grounding-dino-base**: Base GroundingDINO model
/// > - **llmdet-tiny/base/large**: LLMDet models of different sizes
/// > - **mm-gdino-tiny/base/large**: Multi-modal GroundingDINO variants
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Open-Set Detection**: Detect objects described by text
/// > - [X] **Language Guidance**: Use text prompts to guide detection
/// > - [X] **Multi-Dataset Training**: Trained on diverse datasets
/// > - [X] **Flexible Architecture**: Various model sizes and capabilities
/// >
/// Model configuration for `GroundingDino` Series
///
impl crate::Config {
    /// Base configuration for GroundingDino series models
    pub fn grounding_dino() -> Self {
        Self::default()
            .with_name("grounding-dino")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 800)
            .with_model_ixx(0, 3, 1066)
            .with_resize_mode_type(crate::ResizeModeType::FitAdaptive)
            .with_resize_filter(crate::ResizeFilter::CatmullRom)
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
            .with_tokenizer_file("grounding-dino/tokenizer.json")
            .with_config_file("grounding-dino/config.json")
            .with_special_tokens_map_file("grounding-dino/special_tokens_map.json")
            .with_tokenizer_config_file("grounding-dino/tokenizer_config.json")
    }

    /// Tiny GroundingDINO model
    pub fn grounding_dino_tiny() -> Self {
        Self::grounding_dino().with_model_file("gdino-tiny.onnx")
    }

    /// Base GroundingDINO model
    pub fn grounding_dino_base() -> Self {
        Self::grounding_dino().with_model_file("gdino-base.onnx")
    }

    /// LLMDet tiny model
    pub fn llmdet_tiny() -> Self {
        Self::grounding_dino().with_model_file("llmdet-tiny.onnx")
    }

    /// LLMDet base model
    pub fn llmdet_base() -> Self {
        Self::grounding_dino().with_model_file("llmdet-base.onnx")
    }

    /// LLMDet large model
    pub fn llmdet_large() -> Self {
        Self::grounding_dino().with_model_file("llmdet-large.onnx")
    }

    /// mm-GroundingDINO tiny with Objects365v1 + GoldG + GRiT + v3det
    pub fn mm_gdino_tiny_o365v1_goldg_grit_v3det() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-tiny-o365v1-goldg-grit-v3det.onnx")
    }

    /// mm-GroundingDINO tiny with Objects365v1 + GoldG + GRiT
    pub fn mm_gdino_tiny_o365v1_goldg_grit() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-tiny-o365v1-goldg-grit.onnx")
    }

    /// mm-GroundingDINO tiny with Objects365v1 + GoldG + v3det
    pub fn mm_gdino_tiny_o365v1_goldg_v3det() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-tiny-o365v1-goldg-v3det.onnx")
    }

    /// mm-GroundingDINO tiny with Objects365v1 + GoldG
    pub fn mm_gdino_tiny_o365v1_goldg() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-tiny-o365v1-goldg.onnx")
    }

    /// mm-GroundingDINO base with Objects365v1 + GoldG + v3det
    pub fn mm_gdino_base_o365v1_goldg_v3det() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-base-o365v1-goldg-v3det.onnx")
    }

    /// mm-GroundingDINO base with all datasets
    pub fn mm_gdino_base_all() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-base-all.onnx")
    }

    /// mm-GroundingDINO large with Objects365v2 + OIv6 + GoldG
    pub fn mm_gdino_large_o365v2_oiv6_goldg() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-large-o365v2-oiv6-goldg.onnx")
    }

    /// mm-GroundingDINO large with all datasets
    pub fn mm_gdino_large_all() -> Self {
        Self::grounding_dino().with_model_file("mm-gdino-large-all.onnx")
    }
}
