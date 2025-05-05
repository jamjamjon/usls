use crate::{impl_model_config_methods, impl_process_config_methods};

/// Model configuration for `BLIP`
impl crate::Options {
    pub fn blip() -> Self {
        Self::default().with_model_name("blip").with_batch_size(1)
    }

    #[allow(clippy::excessive_precision)]
    pub fn blip_visual() -> Self {
        Self::blip()
            .with_model_kind(crate::Kind::Vision)
            .with_model_ixx(0, 2, 384.into())
            .with_model_ixx(0, 3, 384.into())
            .with_image_mean(&[0.48145466, 0.4578275, 0.40821073])
            .with_image_std(&[0.26862954, 0.26130258, 0.27577711])
            .with_resize_filter("Bilinear")
            .with_normalize(true)
    }

    pub fn blip_textual() -> Self {
        Self::blip().with_model_kind(crate::Kind::Language)
    }

    pub fn blip_v1_base_caption_visual() -> Self {
        Self::blip_visual()
            .with_model_version(1.into())
            .with_model_file("v1-base-caption-visual.onnx")
    }

    pub fn blip_v1_base_caption_textual() -> Self {
        Self::blip_textual()
            .with_model_version(1.into())
            .with_model_file("v1-base-caption-textual.onnx")
    }
}

#[derive(Debug, Clone)]
pub struct BLIPConfig {
    pub visual: crate::ModelConfig,
    pub textual: crate::ModelConfig,
    pub processor: crate::ProcessorConfig,
}

impl Default for BLIPConfig {
    fn default() -> Self {
        Self {
            visual: crate::ModelConfig::default()
                .with_name("blip")
                // .with_kind(Kind::Vision)
                .with_ixx(0, 0, 1.into())
                .with_ixx(0, 1, 3.into())
                .with_ixx(0, 2, 384.into())
                .with_ixx(0, 3, 384.into()),
            textual: crate::ModelConfig::default()
                .with_name("blip")
                // .with_kind(Kind::Language)
                .with_ixx(0, 0, 1.into())
                .with_ixx(0, 1, 3.into()),
            processor: crate::ProcessorConfig::default()
                .with_tokenizer_file("blip/tokenizer.json")
                .with_config_file("blip/config.json")
                .with_special_tokens_map_file("blip/special_tokens_map.json")
                .with_tokenizer_config_file("blip/tokenizer_config.json")
                .with_image_mean(&[0.48145466, 0.4578275, 0.40821073])
                .with_image_std(&[0.26862954, 0.2613026, 0.2757771]),
        }
    }
}

impl_model_config_methods!(BLIPConfig, visual);
impl_model_config_methods!(BLIPConfig, textual);
impl_process_config_methods!(BLIPConfig, processor);

impl BLIPConfig {
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

    pub fn blip_v1_base_caption() -> Self {
        Self::default()
            .with_textual_file("v1-base-caption-textual.onnx")
            .with_visual_file("v1-base-caption-visual.onnx")
    }
}
