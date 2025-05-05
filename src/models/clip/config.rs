use crate::{impl_model_config_methods, impl_process_config_methods, Kind};

/// Model configuration for `CLIP`
impl crate::Options {
    pub fn clip() -> Self {
        Self::default()
            .with_model_name("clip")
            .with_model_ixx(0, 0, 1.into())
            .with_model_num_dry_run(3)
    }

    pub fn clip_visual() -> Self {
        Self::clip()
            .with_model_kind(Kind::Vision)
            .with_model_ixx(0, 2, 224.into())
            .with_model_ixx(0, 3, 224.into())
            .with_image_mean(&[0.48145466, 0.4578275, 0.40821073])
            .with_image_std(&[0.26862954, 0.2613026, 0.2757771])
    }

    pub fn clip_textual() -> Self {
        Self::clip()
            .with_model_kind(Kind::Language)
            .with_model_max_length(77)
    }

    pub fn clip_vit_b16_visual() -> Self {
        Self::clip_visual().with_model_file("vit-b16-visual.onnx")
    }

    pub fn clip_vit_b16_textual() -> Self {
        Self::clip_textual().with_model_file("vit-b16-textual.onnx")
    }

    pub fn clip_vit_b32_visual() -> Self {
        Self::clip_visual().with_model_file("vit-b32-visual.onnx")
    }

    pub fn clip_vit_b32_textual() -> Self {
        Self::clip_textual().with_model_file("vit-b32-textual.onnx")
    }

    pub fn clip_vit_l14_visual() -> Self {
        Self::clip_visual().with_model_file("vit-l14-visual.onnx")
    }

    pub fn clip_vit_l14_textual() -> Self {
        Self::clip_textual().with_model_file("vit-l14-textual.onnx")
    }

    pub fn jina_clip_v1() -> Self {
        Self::default()
            .with_model_name("jina-clip-v1")
            .with_model_ixx(0, 0, 1.into())
    }

    pub fn jina_clip_v1_visual() -> Self {
        Self::jina_clip_v1()
            .with_model_kind(Kind::Vision)
            .with_model_ixx(0, 2, 224.into())
            .with_model_ixx(0, 3, 224.into())
            .with_image_mean(&[0.48145466, 0.4578275, 0.40821073])
            .with_image_std(&[0.26862954, 0.2613026, 0.2757771])
            .with_model_file("visual.onnx")
    }

    pub fn jina_clip_v1_textual() -> Self {
        Self::jina_clip_v1()
            .with_model_kind(Kind::Language)
            .with_model_file("textual.onnx")
    }
}

#[derive(Debug, Clone)]
pub struct CLIPConfig {
    pub visual: crate::ModelConfig,
    pub textual: crate::ModelConfig,
    pub processor: crate::ProcessorConfig,
}

impl Default for CLIPConfig {
    fn default() -> Self {
        Self {
            visual: crate::ModelConfig::default()
                .with_name("clip")
                .with_kind(Kind::Vision)
                .with_ixx(0, 0, 1.into())
                .with_ixx(0, 1, 3.into())
                .with_ixx(0, 2, 224.into())
                .with_ixx(0, 3, 224.into()),
            textual: crate::ModelConfig::default()
                .with_name("clip")
                .with_kind(Kind::Language)
                .with_ixx(0, 0, 1.into())
                .with_ixx(0, 1, 3.into()),
            processor: crate::ProcessorConfig::default()
                .with_model_max_length(77)
                .with_tokenizer_file("clip/tokenizer.json")
                .with_config_file("clip/config.json")
                .with_special_tokens_map_file("clip/special_tokens_map.json")
                .with_tokenizer_config_file("clip/tokenizer_config.json")
                .with_image_mean(&[0.48145466, 0.4578275, 0.40821073])
                .with_image_std(&[0.26862954, 0.2613026, 0.2757771]),
        }
    }
}

impl_model_config_methods!(CLIPConfig, visual);
impl_model_config_methods!(CLIPConfig, textual);
impl_process_config_methods!(CLIPConfig, processor);

impl CLIPConfig {
    // TODO
    pub fn with_model_device(mut self, device: crate::Device) -> Self {
        self.visual = self.visual.with_device(device);
        self.textual = self.textual.with_device(device);
        self
    }

    // TODO
    pub fn with_model_dtype(mut self, dtype: crate::DType) -> Self {
        self.visual = self.visual.with_dtype(dtype);
        self.textual = self.textual.with_dtype(dtype);
        self
    }

    // TODO
    pub fn with_model_name(mut self, name: &'static str) -> Self {
        self.visual = self.visual.with_name(name);
        self.textual = self.textual.with_name(name);
        self
    }

    pub fn clip_vit_b16() -> Self {
        Self::default()
            .with_textual_file("vit-b16-textual.onnx")
            .with_visual_file("vit-b16-visual.onnx")
    }
    pub fn clip_vit_b32() -> Self {
        Self::default()
            .with_textual_file("vit-b32-textual.onnx")
            .with_visual_file("vit-b32-visual.onnx")
    }

    pub fn clip_vit_l14() -> Self {
        Self::default()
            .with_textual_file("vit-l14-textual.onnx")
            .with_visual_file("vit-l14-visual.onnx")
    }

    pub fn jina_clip_v1() -> Self {
        Self::default()
            // .with_visual_name("jina-clip-v1")
            // .with_textual_name("jina-clip-v1")
            .with_model_name("jina-clip-v1")
            .with_visual_file("visual.onnx")
            .with_textual_file("textual.onnx")
            .with_tokenizer_file("jina-clip-v1/tokenizer.json")
            .with_config_file("jina-clip-v1/config.json")
            .with_special_tokens_map_file("jina-clip-v1/special_tokens_map.json")
            .with_tokenizer_config_file("jina-clip-v1/tokenizer_config.json")
    }
}
