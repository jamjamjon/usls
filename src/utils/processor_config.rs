use aksr::Builder;
use anyhow::Result;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::{Hub, ResizeMode};

#[derive(Builder, Debug, Clone)]
pub struct ProcessorConfig {
    // Vision
    pub image_width: Option<u32>,
    pub image_height: Option<u32>,
    pub resize_mode: ResizeMode,
    pub resize_filter: Option<&'static str>,
    pub padding_value: u8,
    pub normalize: bool,
    pub image_std: Vec<f32>,
    pub image_mean: Vec<f32>,
    pub nchw: bool,
    pub unsigned: bool,

    // Text
    pub model_max_length: Option<u64>,
    pub tokenizer_file: Option<String>,
    pub config_file: Option<String>,
    pub special_tokens_map_file: Option<String>,
    pub tokenizer_config_file: Option<String>,
    pub generation_config_file: Option<String>,
    pub vocab_file: Option<String>,
    pub vocab_txt: Option<String>,
    pub temperature: f32,
    pub topp: f32,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            image_width: None,
            image_height: None,
            resize_mode: ResizeMode::FitExact,
            resize_filter: Some("Bilinear"),
            padding_value: 114,
            normalize: true,
            image_std: vec![],
            image_mean: vec![],
            nchw: true,
            unsigned: false,
            model_max_length: None,
            tokenizer_file: None,
            config_file: None,
            special_tokens_map_file: None,
            tokenizer_config_file: None,
            generation_config_file: None,
            vocab_file: None,
            vocab_txt: None,
            temperature: 1.0,
            topp: 0.9,
        }
    }
}

impl ProcessorConfig {
    pub fn try_build_tokenizer(&self) -> Result<Option<Tokenizer>> {
        let mut hub = Hub::default();

        // tokenizer file
        let mut tokenizer: Tokenizer = match &self.tokenizer_file {
            None => return Ok(None),
            Some(file) => Tokenizer::from_file(hub.try_fetch(file)?)
                .map_err(|err| anyhow::anyhow!("Faild to build tokenizer: {err}"))?,
        };

        // config file
        // TODO: save configs?
        let pad_id = match &self.tokenizer_config_file {
            None => 0u32,
            Some(file) => match hub.try_fetch(file) {
                Ok(x) => {
                    let config: serde_json::Value =
                        serde_json::from_str(&std::fs::read_to_string(x)?)?;
                    config["pad_token_id"].as_u64().unwrap_or(0) as u32
                }
                Err(_err) => 0u32,
            },
        };

        // tokenizer_config file
        let mut max_length = None;
        let mut pad_token = String::from("[PAD]");

        if let Some(file) = &self.tokenizer_config_file {
            match hub.try_fetch(file) {
                Err(_) => {}
                Ok(x) => {
                    let tokenizer_config: serde_json::Value =
                        serde_json::from_str(&std::fs::read_to_string(x)?)?;
                    max_length = tokenizer_config["model_max_length"].as_u64();
                    pad_token = tokenizer_config["pad_token"]
                        .as_str()
                        .unwrap_or("[PAD]")
                        .to_string();
                }
            }
        }

        // TODO: padding
        // if `max_length` specified: use `Fixed` strategy
        // else: use `BatchLongest` strategy
        // TODO: if sequence_length is dynamic, `BatchLongest` is fine
        let tokenizer = match self.model_max_length {
            Some(n) => {
                let n = match max_length {
                    None => n,
                    Some(x) => x.min(n),
                };
                tokenizer
                    .with_padding(Some(PaddingParams {
                        strategy: PaddingStrategy::Fixed(n as _),
                        pad_token,
                        pad_id,
                        ..Default::default()
                    }))
                    .clone()
            }
            None => match max_length {
                Some(n) => tokenizer
                    .with_padding(Some(PaddingParams {
                        strategy: PaddingStrategy::BatchLongest,
                        pad_token,
                        pad_id,
                        ..Default::default()
                    }))
                    .with_truncation(Some(TruncationParams {
                        max_length: n as _,
                        ..Default::default()
                    }))
                    .map_err(|err| anyhow::anyhow!("Failed to truncate: {}", err))?
                    .clone(),
                None => tokenizer
                    .with_padding(Some(PaddingParams {
                        strategy: PaddingStrategy::BatchLongest,
                        pad_token,
                        pad_id,
                        ..Default::default()
                    }))
                    .clone(),
            },
        };

        Ok(Some(tokenizer.into()))
    }
}

#[macro_export]
macro_rules! impl_process_config_methods {
    ($ty:ty, $field:ident) => {
        impl $ty {
            pub fn with_image_width(mut self, image_width: u32) -> Self {
                self.$field = self.$field.with_image_width(image_width);
                self
            }
            pub fn with_image_height(mut self, image_height: u32) -> Self {
                self.$field = self.$field.with_image_height(image_height);
                self
            }
            pub fn with_resize_mode(mut self, resize_mode: $crate::ResizeMode) -> Self {
                self.$field = self.$field.with_resize_mode(resize_mode);
                self
            }
            pub fn with_resize_filter(mut self, resize_filter: &'static str) -> Self {
                self.$field = self.$field.with_resize_filter(resize_filter);
                self
            }
            pub fn with_padding_value(mut self, padding_value: u8) -> Self {
                self.$field = self.$field.with_padding_value(padding_value);
                self
            }
            pub fn with_normalize(mut self, normalize: bool) -> Self {
                self.$field = self.$field.with_normalize(normalize);
                self
            }
            pub fn with_image_std(mut self, image_std: &[f32]) -> Self {
                self.$field = self.$field.with_image_std(image_std);
                self
            }
            pub fn with_image_mean(mut self, image_mean: &[f32]) -> Self {
                self.$field = self.$field.with_image_mean(image_mean);
                self
            }
            pub fn with_nchw(mut self, nchw: bool) -> Self {
                self.$field = self.$field.with_nchw(nchw);
                self
            }
            pub fn with_unsigned(mut self, unsigned: bool) -> Self {
                self.$field = self.$field.with_unsigned(unsigned);
                self
            }
            pub fn with_model_max_length(mut self, model_max_length: u64) -> Self {
                self.$field = self.$field.with_model_max_length(model_max_length);
                self
            }
            pub fn with_tokenizer_file(mut self, tokenizer_file: &str) -> Self {
                self.$field = self.$field.with_tokenizer_file(tokenizer_file);
                self
            }
            pub fn with_config_file(mut self, config_file: &str) -> Self {
                self.$field = self.$field.with_config_file(config_file);
                self
            }
            pub fn with_special_tokens_map_file(mut self, special_tokens_map_file: &str) -> Self {
                self.$field = self
                    .$field
                    .with_special_tokens_map_file(special_tokens_map_file);
                self
            }
            pub fn with_tokenizer_config_file(mut self, tokenizer_config_file: &str) -> Self {
                self.$field = self
                    .$field
                    .with_tokenizer_config_file(tokenizer_config_file);
                self
            }
            pub fn with_generation_config_file(mut self, generation_config_file: &str) -> Self {
                self.$field = self
                    .$field
                    .with_generation_config_file(generation_config_file);
                self
            }
            pub fn with_vocab_file(mut self, vocab_file: &str) -> Self {
                self.$field = self.$field.with_vocab_file(vocab_file);
                self
            }
            pub fn with_vocab_txt(mut self, vocab_txt: &str) -> Self {
                self.$field = self.$field.with_vocab_txt(vocab_txt);
                self
            }
            pub fn with_temperature(mut self, temperature: f32) -> Self {
                self.$field = self.$field.with_temperature(temperature);
                self
            }
            pub fn with_topp(mut self, topp: f32) -> Self {
                self.$field = self.$field.with_topp(topp);
                self
            }
        }
    };
}
