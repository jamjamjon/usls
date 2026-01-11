//! Text processor configuration module.
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// Text processor configuration.
///
/// Contains all settings for text processing including tokenization and generation parameters.
#[derive(aksr::Builder, Debug, Clone, PartialEq)]
pub struct TextProcessorConfig {
    /// Path to tokenizer file.
    pub tokenizer_file: Option<String>,
    /// Path to model configuration file.
    pub config_file: Option<String>,
    /// Path to special tokens mapping file.
    pub special_tokens_map_file: Option<String>,
    /// Path to tokenizer configuration file.
    pub tokenizer_config_file: Option<String>,
    /// Path to generation configuration file.
    pub generation_config_file: Option<String>,
    /// Path to vocabulary file.
    pub vocab_file: Option<String>,
    /// Maximum sequence length for tokenization.
    pub model_max_length: Option<u64>,
    /// Temperature parameter for text generation.
    pub temperature: f32,
    /// Top-p parameter for nucleus sampling.
    pub topp: f32,
}

impl Default for TextProcessorConfig {
    fn default() -> Self {
        Self {
            model_max_length: None,
            tokenizer_file: None,
            config_file: None,
            special_tokens_map_file: None,
            tokenizer_config_file: None,
            generation_config_file: None,
            vocab_file: None,
            temperature: 1.0,
            topp: 0.9,
        }
    }
}

impl TextProcessorConfig {
    // TODO
    /// Build tokenizer from configuration.
    pub fn try_build_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let mut hub = crate::Hub::default();
        let mut tokenizer: Tokenizer = match &self.tokenizer_file {
            None => return Err(anyhow::anyhow!("tokenizer_file is required")),
            Some(file) => Tokenizer::from_file(hub.try_fetch(file)?)
                .map_err(|err| anyhow::anyhow!("Failed to build tokenizer: {err}"))?,
        };

        let pad_id = match &self.tokenizer_config_file {
            None => 0u32,
            Some(file) => match hub.try_fetch(file) {
                Ok(x) => {
                    let config: serde_json::Value =
                        serde_json::from_str(&std::fs::read_to_string(x)?)?;
                    config["pad_token_id"].as_u64().unwrap_or(0) as u32
                }
                Err(_) => 0u32,
            },
        };

        let mut max_length = None;
        let mut pad_token = String::from("[PAD]");

        if let Some(file) = &self.tokenizer_config_file {
            if let Ok(x) = hub.try_fetch(file) {
                let tokenizer_config: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(x)?)?;
                max_length = tokenizer_config["model_max_length"].as_u64();
                pad_token = tokenizer_config["pad_token"]
                    .as_str()
                    .unwrap_or("[PAD]")
                    .to_string();
            }
        }

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

        Ok(tokenizer.into())
    }
}
