use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{
    elapsed_module, ort_inputs, Config, Engine, FromConfig, Image, ImageProcessor, LogitsSampler,
    Scale, TextProcessor, X, Y,
};

#[derive(Debug, Builder)]
pub struct FastVLM {
    vision: Engine,
    text_embed: Engine,
    decoder: Engine,
    scale: Scale,
    image_token: String,
    bos_token: String,
    eos_token: String,
    bos_token_id: u32,
    eos_token_id: u32,
    image_token_id: u32,
    max_length: usize,
    num_hidden_layers: usize,
    head_dim: usize,
    num_key_value_heads: usize,
    num_attention_heads: usize,
    hidden_size: usize,
    batch: usize,
    width: usize,
    height: usize,
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
}

impl FastVLM {
    pub fn new(mut config: Config) -> Result<Self> {
        let vision = Engine::from_config(config.take_module(&crate::Module::Visual)?)?;
        let text_embed = Engine::from_config(config.take_module(&crate::Module::Textual)?)?;
        let decoder =
            Engine::from_config(config.take_module(&crate::Module::TextualDecoderMerged)?)?;
        let max_length = config.inference.max_tokens.unwrap_or(1024);
        let image_token = "<image>".to_string();
        let image_token_id = 151646;
        let eos_token = "<|im_end|>".to_string();
        let eos_token_id = 151645;
        let bos_token = "<|im_start|>".to_string();
        let bos_token_id = 151644;
        let scale = config
            .scale
            .take()
            .ok_or_else(|| anyhow::anyhow!("Scale configuration is required for FastVLM model"))?;
        let (num_hidden_layers, head_dim, num_key_value_heads, hidden_size, num_attention_heads) =
            match &scale {
                Scale::Billion(0.5) => (24, 64, 2, 896, 14),
                _ => unimplemented!(),
            };
        let (batch, height, width) = (
            vision.batch().opt(),
            vision.try_height().unwrap_or(&1024.into()).opt(),
            vision.try_width().unwrap_or(&1024.into()).opt(),
        );
        let image_processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        #[cfg(feature = "vlm")]
        let text_processor = TextProcessor::from_config(config.text_processor)?;
        #[cfg(not(feature = "vlm"))]
        let text_processor = TextProcessor::default();

        Ok(Self {
            vision,
            text_embed,
            decoder,
            scale,
            max_length,
            eos_token_id,
            bos_token_id,
            image_token,
            image_token_id,
            num_hidden_layers,
            head_dim,
            num_key_value_heads,
            num_attention_heads,
            hidden_size,
            bos_token,
            eos_token,
            batch,
            height,
            width,
            image_processor,
            text_processor,
        })
    }

    fn encode_images(&mut self, xs: &[Image]) -> Result<X> {
        let ys = self.image_processor.process(xs)?;
        self.batch = xs.len();
        let output = self.vision.run(ort_inputs![ys]?)?;
        Ok(X::from(output.get::<f32>(0)?))
    }

    pub fn forward(&mut self, images: &[Image], text: &str) -> Result<Vec<Y>> {
        let image_embeddings =
            elapsed_module!("FastVLM", "encode-images", self.encode_images(images)?);
        elapsed_module!(
            "FastVLM",
            "generate",
            self.generate(&image_embeddings, text)
        )
    }

    fn generate(&mut self, image_embeddings: &X, text: &str) -> Result<Vec<Y>> {
        let image_seq_len = image_embeddings.dims()[1];

        // prompt
        let prompt = format!("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{}<|im_end|>\n<|im_start|>assistant\n", text);
        let input_ids: Vec<f32> = self.text_processor.encode_text_ids(&prompt, true)?;

        // inputs embeds
        let input_ids_x = X::from(input_ids.clone())
            .insert_axis(0)?
            .repeat(0, self.batch)?;
        let inputs_embeds0 = {
            let ys = self.text_embed.run(ort_inputs![input_ids_x]?)?;
            X::from(ys.get::<f32>(0)?)
        };
        let text_seq_len = inputs_embeds0.dims()[1];
        // total_sequence_length
        let mut seq_len = text_seq_len + image_seq_len;

        // merge
        let mut inputs_embeds = X::zeros(&[self.batch, seq_len, self.hidden_size]);
        let pos = input_ids
            .iter()
            .position(|&x| x == self.image_token_id as f32)
            .unwrap_or(input_ids.len() / 2);
        inputs_embeds
            .slice_mut(s![.., ..pos, ..])
            .assign(&inputs_embeds0.slice(s![.., ..pos, ..]));
        inputs_embeds
            .slice_mut(s![.., pos..pos + image_seq_len, ..])
            .assign(&image_embeddings.slice(s![.., ..image_seq_len, ..]));
        if pos < text_seq_len {
            inputs_embeds
                .slice_mut(s![.., pos + image_seq_len.., ..])
                .assign(&inputs_embeds0.slice(s![.., pos.., ..]));
        }

        // position ids
        let mut position_ids = X::from((0..seq_len).map(|x| x as f32).collect::<Vec<f32>>())
            .insert_axis(0)?
            .repeat(0, self.batch)?;

        // past key_values
        let mut past_key_values =
            vec![
                X::zeros(&[self.batch, self.num_key_value_heads, 0, self.head_dim]);
                self.num_hidden_layers * 2
            ];

        // token ids
        let mut token_ids: Vec<Vec<u32>> = vec![vec![]; self.batch];
        let mut finished = vec![false; self.batch];
        let mut last_tokens: Vec<f32> = vec![0.; self.batch];

        // decode
        let logits_sampler = LogitsSampler::new();

        for _ in 0..self.max_length {
            // inputs
            let attention_mask = X::ones(&[self.batch, seq_len]);
            let mut xs = vec![inputs_embeds.clone(), attention_mask, position_ids.clone()];
            for i in 0..self.num_hidden_layers {
                xs.push(past_key_values[i * 2].clone());
                xs.push(past_key_values[i * 2 + 1].clone());
            }

            // decoder
            let (logits, new_past_key_values) = {
                let decoder_outputs = self.decoder.run(&xs)?;
                let logits = X::from(decoder_outputs.get::<f32>(0)?);
                let new_past_key_values: Vec<X> = (1..decoder_outputs.len())
                    .step_by(2)
                    .flat_map(|i| [i, i + 1])
                    .map(|i| X::from(decoder_outputs.get::<f32>(i).unwrap()))
                    .collect();
                (logits, new_past_key_values)
            };
            past_key_values = new_past_key_values;

            // decode each token for each batch
            for (batch_idx, logit) in logits.axis_iter(Axis(0)).enumerate() {
                if !finished[batch_idx] {
                    let token_id = logits_sampler.decode(
                        &logit
                            .slice(s![-1, ..])
                            .into_owned()
                            .into_raw_vec_and_offset()
                            .0,
                    )?;

                    // early return
                    if token_id == self.eos_token_id {
                        finished[batch_idx] = true;
                    } else {
                        token_ids[batch_idx].push(token_id);
                        last_tokens[batch_idx] = token_id as f32;
                    }
                }
            }

            // all finished?
            if finished.iter().all(|&x| x) {
                break;
            }

            // inputs embeds for next iteration
            let input_ids_x = X::from(last_tokens.clone()).insert_axis(1)?;
            inputs_embeds = {
                let ys = self.text_embed.run(ort_inputs![input_ids_x]?)?;
                X::from(ys.get::<f32>(0)?)
            };

            // update position_ids for next iteration
            position_ids = X::from(
                position_ids
                    .slice(s![.., -1..])
                    .mapv(|x| x + 1.0)
                    .into_owned()
                    .into_dyn(),
            );
            seq_len += 1;
        }

        // decode
        let texts = self.text_processor.decode_tokens_batch(&token_ids, true)?;
        let texts = texts
            .into_par_iter()
            .map(|x| Y::default().with_texts(&[x.into()]))
            .collect::<Vec<_>>();

        Ok(texts)
    }
}
