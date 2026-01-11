//! Text processing pipeline with tokenization and vocabulary support.

use aksr::Builder;
use anyhow::Result;

use crate::{ChatTemplate, FromConfig, LogitsSampler, TextProcessorConfig};

/// Manages tokenization and sampling for Vision-Language Models (VLMs).
///
/// # 📝 TextProcessor
///
/// A comprehensive text processing pipeline that handles tokenization, encoding/decoding,
/// and logits sampling for conversational AI models. Provides fast tokenization using
/// the `tokenizers` library and configurable sampling strategies.
///
/// ## Features
///
/// - **Tokenizer**: Fast tokenization using the `tokenizers` library with support for various vocabularies
/// - **LogitsSampler**: Configurable sampling strategies (Greedy, Top-P, Temperature)
/// - **ChatTemplate**: Standardized input formatting for conversational models
/// - **Batch Processing**: Efficient batch tokenization and decoding operations
/// - **Error Handling**: Comprehensive error reporting with context
///
/// ## Supported Operations
///
/// - **Encoding**: Text to token IDs, tokens, and batch operations
/// - **Decoding**: Token IDs back to text with batch support
/// - **Sampling**: Configurable text generation strategies
/// - **Templates**: Chat formatting for conversational models
///
/// ## Examples
///
/// ```no_run
/// use usls::{TextProcessor, TextProcessorConfig};
///
/// let config = TextProcessorConfig::default()
///     .with_model_path("./models/tokenizer.json")
///     .with_temperature(0.7)
///     .with_topp(0.9);
///
/// let processor = TextProcessor::from_config(config)?;
///
/// // Encode text
/// let encoding = processor.encode_text("Hello, world!", false)?;
/// let ids = processor.encode_text_ids("Hello, world!", false)?;
///
/// // Decode tokens
/// let text = processor.decode_tokens(&[1, 2, 3], false)?;
/// ```
///
#[derive(Builder, Debug)]
pub struct TextProcessor {
    pub tokenizer: tokenizers::Tokenizer,
    pub logits_sampler: LogitsSampler,
    pub chat_template: ChatTemplate,
}

impl FromConfig for TextProcessor {
    type Config = TextProcessorConfig;

    fn from_config(config: TextProcessorConfig) -> Result<Self> {
        let logits_sampler = LogitsSampler::new()
            .with_temperature(config.temperature)
            .with_topp(config.topp);

        let tokenizer = config.try_build_tokenizer()?;
        let chat_template = ChatTemplate;

        Ok(Self {
            tokenizer,
            logits_sampler,
            chat_template,
        })
    }
}

impl TextProcessor {
    /// Encode a single text string to tokenization encoding.
    pub fn encode_text(&self, x: &str, skip_special_tokens: bool) -> Result<tokenizers::Encoding> {
        self.tokenizer
            .encode(x, skip_special_tokens)
            .map_err(|err| {
                anyhow::anyhow!(
                    "Failed to encode text '{}': {}",
                    x.chars().take(50).collect::<String>(),
                    err
                )
            })
    }

    /// Encode multiple text strings in batch.
    pub fn encode_texts(
        &self,
        xs: &[&str],
        skip_special_tokens: bool,
    ) -> Result<Vec<tokenizers::Encoding>> {
        self.tokenizer
            .encode_batch(xs.to_vec(), skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("Failed to encode batch of {} texts: {}", xs.len(), err))
    }

    /// Encode text to token IDs as f32 vector.
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

    /// Encode multiple texts to token ID vectors in batch.
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

    /// Encode text to token strings.
    pub fn encode_text_tokens(&self, x: &str, skip_special_tokens: bool) -> Result<Vec<String>> {
        Ok(self
            .encode_text(x, skip_special_tokens)?
            .get_tokens()
            .to_vec())
    }

    /// Encode multiple texts to token strings in batch.
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

    /// Decode token IDs back to text string.
    pub fn decode_tokens(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("Failed to decode {} token IDs: {}", ids.len(), err))
    }

    /// Decode batch of token sequences to text strings.
    pub fn decode_tokens_batch2(
        &self,
        ids: &[&[u32]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        self.tokenizer
            .decode_batch(ids, skip_special_tokens)
            .map_err(|err| {
                anyhow::anyhow!(
                    "Failed to decode batch of {} token sequences: {}",
                    ids.len(),
                    err
                )
            })
    }

    /// Decode batch of token vectors to text strings.
    pub fn decode_tokens_batch(
        &self,
        ids: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        self.tokenizer
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
