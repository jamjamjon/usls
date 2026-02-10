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
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<u64>,
    /// Whether to ignore the end-of-sequence token.
    pub ignore_eos: bool,
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
            max_tokens: Default::default(),
            ignore_eos: Default::default(),
        }
    }
}

impl TextProcessorConfig {
    /// Build tokenizer from configuration.
    pub fn try_build_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        tracing::debug!("Building tokenizer from config");
        let mut hub = crate::Hub::default();

        // Resolve tokenizer file: check cache first, then fetch
        let mut tokenizer: Tokenizer = match &self.tokenizer_file {
            None => return Err(anyhow::anyhow!("tokenizer_file is required")),
            Some(file) => {
                let path = if let Some(cached) = hub.cached(file) {
                    tracing::debug!("Tokenizer file cache hit: {cached}");
                    cached
                } else {
                    tracing::debug!("Tokenizer file not cached, requesting fetch: {file}");
                    hub.try_fetch(file)?
                };
                Tokenizer::from_file(&path).map_err(|err| {
                    anyhow::anyhow!("Failed to build tokenizer from '{path}': {err}")
                })?
            }
        };

        // TODO
        // Resolve tokenizer config file for pad_id
        let pad_id = match &self.tokenizer_config_file {
            None => 0u32,
            Some(file) => {
                let path = if let Some(cached) = hub.cached(file) {
                    tracing::debug!("Tokenizer config file cache hit: {cached}");
                    Some(cached)
                } else {
                    tracing::debug!("Tokenizer config file not cached, requesting fetch: {file}");
                    hub.try_fetch(file).ok()
                };
                match path {
                    Some(x) => {
                        let config: serde_json::Value =
                            serde_json::from_str(&std::fs::read_to_string(&x)?)?;
                        let id = config["pad_token_id"].as_u64().unwrap_or(0) as u32;
                        tracing::debug!("Resolved pad_token_id: {id}");
                        id
                    }
                    None => 0u32,
                }
            }
        };

        let mut max_length = None;
        let mut pad_token = String::from("[PAD]");

        // TODO
        // Resolve tokenizer config for max_length and pad_token
        if let Some(file) = &self.tokenizer_config_file {
            let path = if let Some(cached) = hub.cached(file) {
                Some(cached)
            } else {
                hub.try_fetch(file).ok()
            };
            if let Some(x) = path {
                let tokenizer_config: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(x)?)?;
                max_length = tokenizer_config["model_max_length"].as_u64();
                pad_token = tokenizer_config["pad_token"]
                    .as_str()
                    .unwrap_or("[PAD]")
                    .to_string();
                tracing::debug!(
                    "Tokenizer config resolved: max_length={:?}, pad_token='{}'",
                    max_length,
                    pad_token
                );
            }
        }

        // TODO
        // Apply padding and truncation
        let tokenizer = match self.model_max_length {
            Some(n) => {
                let n = match max_length {
                    None => n,
                    Some(x) => x.min(n),
                };
                tracing::debug!("Applying fixed padding: length={n}, pad_id={pad_id}");
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
                Some(n) => {
                    tracing::debug!("Applying batch-longest padding with truncation: max_length={n}, pad_id={pad_id}");
                    tokenizer
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
                        .map_err(|err| anyhow::anyhow!("Failed to truncate: {err}"))?
                        .clone()
                }
                None => {
                    tracing::debug!(
                        "Applying batch-longest padding without truncation, pad_id={pad_id}"
                    );
                    tokenizer
                        .with_padding(Some(PaddingParams {
                            strategy: PaddingStrategy::BatchLongest,
                            pad_token,
                            pad_id,
                            ..Default::default()
                        }))
                        .clone()
                }
            },
        };

        tracing::debug!("Tokenizer built successfully");
        Ok(tokenizer.into())
    }
}
