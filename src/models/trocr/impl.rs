use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{elapsed, Config, Engine, Image, LogitsSampler, Processor, Scale, Ts, Xs, X, Y};

#[derive(Debug, Copy, Clone)]
pub enum TrOCRKind {
    Printed,
    HandWritten,
}

impl TryFrom<&str> for TrOCRKind {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "printed" => Ok(Self::Printed),
            "handwritten" | "hand-written" => Ok(Self::HandWritten),
            x => anyhow::bail!("Unsupported TrOCRKind: {}", x),
        }
    }
}

#[derive(Debug, Builder)]
pub struct TrOCR {
    encoder: Engine,
    decoder: Engine,
    decoder_merged: Engine,
    max_length: u32,
    eos_token_id: u32,
    decoder_start_token_id: u32,
    ts: Ts,
    n_kvs: usize,
    processor: Processor,
    batch: usize,
    height: usize,
    width: usize,
}

impl TrOCR {
    pub fn new(config: Config) -> Result<Self> {
        let encoder = Engine::try_from_config(&config.visual)?;
        let decoder = Engine::try_from_config(&config.textual_decoder)?;
        let decoder_merged = Engine::try_from_config(&config.textual_decoder_merged)?;
        let (batch, height, width) = (
            encoder.batch().opt(),
            encoder.try_height().unwrap_or(&384.into()).opt(),
            encoder.try_width().unwrap_or(&384.into()).opt(),
        );
        let ts = Ts::merge(&[encoder.ts(), decoder.ts(), decoder_merged.ts()]);
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
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            encoder,
            decoder,
            decoder_merged,
            max_length,
            ts,
            eos_token_id,
            decoder_start_token_id,
            n_kvs,
            batch,
            width,
            height,
            processor,
        })
    }

    pub fn encode(&mut self, xs: &[Image]) -> Result<X> {
        let ys = self.processor.process_images(xs)?;
        self.batch = xs.len(); // update
        let ys = self.encoder.run(ys.into())?;
        Ok(ys[0].to_owned())
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let encoder_hidden_states = elapsed!("encode", self.ts, { self.encode(xs)? });
        let generated = elapsed!("generate", self.ts, {
            self.generate(&encoder_hidden_states)?
        });
        let ys = elapsed!("decode", self.ts, { self.decode(generated)? });

        Ok(ys)
    }

    fn generate(&mut self, encoder_hidden_states: &X) -> Result<Vec<Vec<u32>>> {
        // input_ids
        let input_ids = X::from(vec![self.decoder_start_token_id as f32])
            .insert_axis(0)?
            .repeat(0, self.batch)?;

        // decoder
        let mut decoder_outputs = self.decoder.run(Xs::from(vec![
            input_ids.clone(),
            encoder_hidden_states.clone(),
        ]))?;

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
            decoder_outputs = self.decoder_merged.run(xs.into())?;
        }

        Ok(token_ids)
    }

    pub fn decode(&self, token_ids: Vec<Vec<u32>>) -> Result<Vec<Y>> {
        // decode
        let texts = self.processor.decode_tokens_batch(&token_ids, false)?;

        // to texts
        let texts = texts
            .into_par_iter()
            .map(|x| Y::default().with_texts(&[&x]))
            .collect::<Vec<_>>();

        Ok(texts)
    }

    pub fn summary(&self) {
        self.ts.summary();
    }
}
