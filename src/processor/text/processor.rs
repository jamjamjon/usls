//! Text processing pipeline with tokenization and vocabulary support.

use aksr::Builder;
use anyhow::Result;

use super::LogitsSampler;
use crate::{FromConfig, TextProcessorConfig};

/// Text processing pipeline.
///
/// Handles tokenization, encoding, decoding, and logits sampling.
/// Requires the `tokenizers` feature for full functionality.
#[derive(Builder, Debug, Default)]
pub struct TextProcessor {
    pub tokenizer: Option<tokenizers::Tokenizer>,
    pub logits_sampler: Option<LogitsSampler>,
    // TODO: chat_template
}

impl FromConfig for TextProcessor {
    type Config = TextProcessorConfig;

    fn from_config(config: TextProcessorConfig) -> Result<Self> {
        let logits_sampler = LogitsSampler::new()
            .with_temperature(config.temperature)
            .with_topp(config.topp);

        let tokenizer = config.try_build_tokenizer()?;

        Ok(Self {
            tokenizer,
            logits_sampler: Some(logits_sampler),
        })
    }
}

impl TextProcessor {
    // /// Deprecated: Use `TextProcessor::from_config()` instead.
    // #[deprecated(
    //     since = "0.3.0",
    //     note = "Use TextProcessor::from_config() for zero-copy construction"
    // )]
    // pub fn try_from_config(config: &TextProcessorConfig) -> Result<Self> {
    //     Self::from_config(config.clone())
    // }

    /// Check if tokenizer is available.
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    pub fn encode_text(&self, x: &str, skip_special_tokens: bool) -> Result<tokenizers::Encoding> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "No tokenizer configured in TextProcessor. Please initialize with a tokenizer."
            )
        })?;

        tokenizer.encode(x, skip_special_tokens).map_err(|err| {
            anyhow::anyhow!(
                "Failed to encode text '{}': {}",
                x.chars().take(50).collect::<String>(),
                err
            )
        })
    }

    pub fn encode_texts(
        &self,
        xs: &[&str],
        skip_special_tokens: bool,
    ) -> Result<Vec<tokenizers::Encoding>> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "No tokenizer configured in TextProcessor. Please initialize with a tokenizer."
            )
        })?;

        tokenizer
            .encode_batch(xs.to_vec(), skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("Failed to encode batch of {} texts: {}", xs.len(), err))
    }

    pub fn encode_text_ids(&self, x: &str, skip_special_tokens: bool) -> Result<Vec<f32>> {
        let ids: Vec<f32> = if x.is_empty() {
            vec![0.0f32]
        } else {
            self.encode_text(x, skip_special_tokens)?
                .get_ids()
                .iter()
                .map(|x| *x as f32)
                .collect()
        };

        Ok(ids)
    }

    pub fn encode_texts_ids(
        &self,
        xs: &[&str],
        skip_special_tokens: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let ids: Vec<Vec<f32>> = if xs.is_empty() {
            vec![vec![0.0f32]]
        } else {
            self.encode_texts(xs, skip_special_tokens)?
                .into_iter()
                .map(|encoding| encoding.get_ids().iter().map(|x| *x as f32).collect())
                .collect()
        };

        Ok(ids)
    }

    pub fn encode_text_tokens(&self, x: &str, skip_special_tokens: bool) -> Result<Vec<String>> {
        Ok(self
            .encode_text(x, skip_special_tokens)?
            .get_tokens()
            .to_vec())
    }

    pub fn encode_texts_tokens(
        &self,
        xs: &[&str],
        skip_special_tokens: bool,
    ) -> Result<Vec<Vec<String>>> {
        Ok(self
            .encode_texts(xs, skip_special_tokens)?
            .into_iter()
            .map(|encoding| encoding.get_tokens().to_vec())
            .collect())
    }

    pub fn decode_tokens(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "No tokenizer configured in TextProcessor. Please initialize with a tokenizer."
            )
        })?;

        tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("Failed to decode {} token IDs: {}", ids.len(), err))
    }

    pub fn decode_tokens_batch2(
        &self,
        ids: &[&[u32]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "No tokenizer configured in TextProcessor. Please initialize with a tokenizer."
            )
        })?;

        tokenizer
            .decode_batch(ids, skip_special_tokens)
            .map_err(|err| {
                anyhow::anyhow!(
                    "Failed to decode batch of {} token sequences: {}",
                    ids.len(),
                    err
                )
            })
    }

    pub fn decode_tokens_batch(
        &self,
        ids: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "No tokenizer configured in TextProcessor. Please initialize with a tokenizer."
            )
        })?;

        tokenizer
            .decode_batch(
                &ids.iter().map(|x| x.as_slice()).collect::<Vec<_>>(),
                skip_special_tokens,
            )
            .map_err(|err| {
                anyhow::anyhow!(
                    "Failed to decode batch of {} token vectors: {}",
                    ids.len(),
                    err
                )
            })
    }
}
