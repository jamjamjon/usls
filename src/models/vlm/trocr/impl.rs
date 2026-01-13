use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;
use std::str::FromStr;

use crate::{
    elapsed_module, inputs, Config, Engine, Engines, FromConfig, Image, ImageProcessor,
    LogitsSampler, Model, Module, Scale, TextProcessor, X, Y,
};

/// TrOCR model variants for different text types.
#[derive(Debug, Copy, Clone)]
pub enum TrOCRKind {
    /// Model variant optimized for machine-printed text recognition
    Printed,
    /// Model variant optimized for handwritten text recognition
    HandWritten,
}

impl FromStr for TrOCRKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "printed" => Ok(Self::Printed),
            "handwritten" | "hand-written" => Ok(Self::HandWritten),
            x => anyhow::bail!("Unsupported TrOCRKind: {x}"),
        }
    }
}

/// TrOCR model for optical character recognition.
///
/// TrOCR is a transformer-based OCR model that combines an image encoder with
/// a text decoder for end-to-end text recognition. It supports both printed and
/// handwritten text recognition through different model variants.
///
/// The model consists of:
/// - An encoder that processes the input image
/// - A decoder that generates text tokens
/// - A merged decoder variant for optimized inference
/// - A processor for image preprocessing and text postprocessing
#[derive(Debug, Builder)]
pub struct TrOCR {
    /// Maximum length of generated text sequence
    pub max_length: u32,
    /// Token ID representing end of sequence
    pub eos_token_id: u32,
    /// Token ID used to start text generation
    pub decoder_start_token_id: u32,
    /// Number of key-value pairs in decoder attention
    pub n_kvs: usize,
    /// Image and text processor for pre/post processing
    pub image_processor: ImageProcessor,
    pub text_processor: TextProcessor,
    /// Batch size for inference
    pub batch: usize,
    /// Input image height
    pub height: usize,
    /// Input image width  
    pub width: usize,
    pub spec: String,
}

impl Model for TrOCR {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let encoder = Engine::from_config(config.take_module(&Module::Visual)?)?;
        let decoder = Engine::from_config(config.take_module(&Module::TextualDecoder)?)?;
        let decoder_merged =
            Engine::from_config(config.take_module(&Module::TextualDecoderMerged)?)?;
        let (batch, height, width) = (
            encoder.batch().opt(),
            encoder.try_height().unwrap_or(&384.into()).opt(),
            encoder.try_width().unwrap_or(&384.into()).opt(),
        );

        // "bos_token": "<s>",  "eos_token": "</s>",  "sep_token": "</s>",
        // "model_max_length": 1000000000000000019884624838656,
        // let bos_token = "<s>";
        // let eos_token = "</s>";
        // let sep_token = "</s>";
        // let bos_token_id = 0;
        // let pad_token_id = 1;
        let max_length = 1024; // TODO
        let eos_token_id = 2;
        let decoder_start_token_id = 2;
        let n_kvs = match &config.scale {
            Some(Scale::S) => 6,
            Some(Scale::B) => 12,
            _ => unimplemented!(),
        };
        let image_processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        #[cfg(feature = "vlm")]
        let text_processor = TextProcessor::from_config(config.text_processor)?;
        #[cfg(not(feature = "vlm"))]
        let text_processor = TextProcessor::default();

        let spec = encoder.spec().to_owned();

        let model = Self {
            max_length,
            eos_token_id,
            decoder_start_token_id,
            n_kvs,
            batch,
            width,
            height,
            image_processor,
            text_processor,
            spec,
        };

        let mut engines = Engines::default();
        engines.insert(Module::Visual, encoder);
        engines.insert(Module::TextualDecoder, decoder);
        engines.insert(Module::TextualDecoderMerged, decoder_merged);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let encoder_hidden_states =
            elapsed_module!("TrOCR", "encode", self.encode_internal(engines, images)?);
        let generated = elapsed_module!("TrOCR", "generate", {
            self.generate_internal(engines, &encoder_hidden_states)?
        });
        elapsed_module!("TrOCR", "decode", self.decode_internal(generated))
    }
}

impl TrOCR {
    fn encode_internal(&mut self, engines: &mut Engines, xs: &[Image]) -> Result<X> {
        let ys = self.image_processor.process(xs)?;
        self.batch = xs.len();
        let output = engines.run(&Module::Visual, inputs![ys]?)?;
        let x = output
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get encoder output"))?;
        Ok(X::from(x))
    }

    fn generate_internal(
        &mut self,
        engines: &mut Engines,
        encoder_hidden_states: &X,
    ) -> Result<Vec<Vec<u32>>> {
        // input_ids
        let input_ids = X::from(vec![self.decoder_start_token_id as f32])
            .insert_axis(0)?
            .repeat(0, self.batch)?;

        // decoder
        let mut decoder_outputs = {
            let ys = engines.run(
                &Module::TextualDecoder,
                inputs![input_ids.clone(), encoder_hidden_states.clone()]?,
            )?;
            (0..ys.len())
                .map(|i| X::from(ys.get::<f32>(i).unwrap()))
                .collect::<Vec<_>>()
        };

        // encoder kvs
        let encoder_kvs: Vec<_> = (3..4 * self.n_kvs)
            .step_by(4)
            .flat_map(|i| [i, i + 1])
            .map(|i| decoder_outputs[i].clone())
            .collect();

        // token ids
        let mut token_ids: Vec<Vec<u32>> = vec![vec![]; self.batch];
        let mut finished = vec![false; self.batch];
        let mut last_tokens: Vec<f32> = vec![0.; self.batch];
        let logits_sampler = LogitsSampler::new();

        // generate
        for _ in 0..self.max_length {
            let logits = &decoder_outputs[0];
            let decoder_kvs: Vec<_> = (1..(4 * self.n_kvs) - 2)
                .step_by(4)
                .flat_map(|i| [i, i + 1])
                .map(|i| decoder_outputs[i].clone())
                .collect();

            // decode each token for each batch
            for (i, logit) in logits.axis_iter(Axis(0)).enumerate() {
                if !finished[i] {
                    let token_id = logits_sampler.decode(
                        &logit
                            .slice(s![-1, ..])
                            .into_owned()
                            .into_raw_vec_and_offset()
                            .0,
                    )?;

                    if token_id == self.eos_token_id {
                        finished[i] = true;
                    } else {
                        token_ids[i].push(token_id);
                    }

                    // update
                    last_tokens[i] = token_id as f32;
                }
            }

            // all finished?
            if finished.iter().all(|&x| x) {
                break;
            }

            // build inputs
            let input_ids = X::from(last_tokens.clone()).insert_axis(1)?;
            let mut xs = vec![input_ids, encoder_hidden_states.clone()];
            for i in 0..self.n_kvs {
                xs.push(decoder_kvs[i * 2].clone());
                xs.push(decoder_kvs[i * 2 + 1].clone());
                xs.push(encoder_kvs[i * 2].clone());
                xs.push(encoder_kvs[i * 2 + 1].clone());
            }
            xs.push(X::ones(&[1])); // use_cache

            // generate
            decoder_outputs = {
                let ys = engines.run(&Module::TextualDecoderMerged, &xs)?;
                (0..ys.len())
                    .map(|i| X::from(ys.get::<f32>(i).unwrap()))
                    .collect::<Vec<_>>()
            };
        }

        Ok(token_ids)
    }

    fn decode_internal(&self, token_ids: Vec<Vec<u32>>) -> Result<Vec<Y>> {
        // decode
        let texts = self.text_processor.decode_tokens_batch(&token_ids, false)?;

        // to texts
        let texts = texts
            .into_par_iter()
            .map(|x| Y::default().with_texts(&[x.into()]))
            .collect::<Vec<_>>();

        Ok(texts)
    }
}
