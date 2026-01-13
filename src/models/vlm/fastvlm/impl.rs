use aksr::Builder;
use anyhow::Result;
use ndarray::s;

use crate::{
    inputs, Config, Engine, Engines, FromConfig, Image, ImageProcessor, LogitsSampler, Model,
    Module, Scale, TextProcessor, X, Y,
};

/// FastVLM - A fast Vision-Language Model.
///
/// This model implements the unified `Model` trait with multi-engine support.
/// It uses 3 separate engines:
/// - `Visual`: Image encoder
/// - `Textual`: Text embedding
/// - `TextualDecoderMerged`: Autoregressive decoder
///
/// The engines are managed externally via `Engines` and passed to `run()`.
#[derive(Debug, Builder)]
pub struct FastVLM {
    scale: Scale,
    image_token: String,
    bos_token: String,
    eos_token: String,
    bos_token_id: u32,
    eos_token_id: u32,
    image_token_id: u32,
    max_length: u64,
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
    fn generate_one(&mut self, engines: &mut Engines, image: &Image, text: &str) -> Result<String> {
        // Process single image
        let image_embeddings = self.encode_image(engines, image)?;

        // Generate text for single image
        let batch = 1;
        self.batch = batch;

        // prompt
        let prompt = format!("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{text}<|im_end|>\n<|im_start|>assistant\n");
        let input_ids: Vec<f32> = self.text_processor.encode_text_ids(&prompt, true)?;

        // inputs embeds
        let input_ids_x = X::from(input_ids.clone()).insert_axis(0)?;
        let inputs_embeds0 = {
            let ys = engines.run(&Module::Textual, inputs![input_ids_x]?)?;
            X::from(
                ys.get::<f32>(0)
                    .ok_or_else(|| anyhow::anyhow!("Failed to get text embeddings"))?,
            )
        };
        let text_seq_len = inputs_embeds0.dims()[1];
        let image_seq_len = image_embeddings.dims()[1];

        // total_sequence_length
        let mut seq_len = text_seq_len + image_seq_len;

        // merge
        let mut inputs_embeds = X::zeros(&[batch, seq_len, self.hidden_size]);
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
        let mut position_ids =
            X::from((0..seq_len).map(|x| x as f32).collect::<Vec<f32>>()).insert_axis(0)?;

        // past key_values
        let mut past_key_values =
            vec![
                X::zeros(&[batch, self.num_key_value_heads, 0, self.head_dim]);
                self.num_hidden_layers * 2
            ];

        // token ids
        let mut token_ids: Vec<u32> = vec![];
        let mut finished = false;
        let mut last_token: f32 = 0.;

        // decode
        let logits_sampler = LogitsSampler::new();

        for _ in 0..self.max_length {
            // inputs
            let attention_mask = X::ones(&[batch, seq_len]);
            let mut xs = vec![inputs_embeds.clone(), attention_mask, position_ids.clone()];
            for i in 0..self.num_hidden_layers {
                xs.push(past_key_values[i * 2].clone());
                xs.push(past_key_values[i * 2 + 1].clone());
            }

            // decoder
            let (logits, new_past_key_values) = {
                let decoder_outputs = engines.run(&Module::TextualDecoderMerged, &xs)?;
                let logits = X::from(
                    decoder_outputs
                        .get::<f32>(0)
                        .ok_or_else(|| anyhow::anyhow!("Failed to get logits"))?,
                );
                let new_past_key_values: Vec<X> = (1..decoder_outputs.len())
                    .step_by(2)
                    .flat_map(|i| [i, i + 1])
                    .map(|i| X::from(decoder_outputs.get::<f32>(i).unwrap()))
                    .collect();
                (logits, new_past_key_values)
            };
            past_key_values = new_past_key_values;

            if !finished {
                let token_id = logits_sampler.decode(
                    &logits
                        .slice(s![0, -1, ..])
                        .into_owned()
                        .into_raw_vec_and_offset()
                        .0,
                )?;

                // early return
                if token_id == self.eos_token_id {
                    finished = true;
                } else {
                    token_ids.push(token_id);
                    last_token = token_id as f32;
                }
            }

            // all finished?
            if finished {
                break;
            }

            // inputs embeds for next iteration
            let input_ids_x = X::from(vec![last_token]).insert_axis(1)?;
            inputs_embeds = {
                let ys = engines.run(&Module::Textual, inputs![input_ids_x]?)?;
                X::from(
                    ys.get::<f32>(0)
                        .ok_or_else(|| anyhow::anyhow!("Failed to get next token embeddings"))?,
                )
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

        // decode tokens
        let text = self.text_processor.decode_tokens(&token_ids, true)?;

        Ok(text)
    }

    fn encode_image(&mut self, engines: &mut Engines, image: &Image) -> Result<X> {
        let xs = self.image_processor.process(&[image.clone()])?;
        let output = engines.run(&Module::Visual, inputs![xs]?)?;
        let x = output
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get vision output"))?;
        Ok(X::from(x))
    }
}

/// Implement the Model trait for FastVLM.
impl Model for FastVLM {
    type Input<'a> = (&'a [Image], &'a str);

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        "fastvlm"
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let vision = Engine::from_config(config.take_module(&Module::Visual)?)?;
        let text_embed = Engine::from_config(config.take_module(&Module::Textual)?)?;
        let decoder = Engine::from_config(config.take_module(&Module::TextualDecoderMerged)?)?;

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
        let text_processor = TextProcessor::from_config(config.text_processor)?;

        let model = Self {
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
        };

        // Build engines collection
        let mut engines = Engines::new();
        engines.insert(Module::Visual, vision);
        engines.insert(Module::Textual, text_embed);
        engines.insert(Module::TextualDecoderMerged, decoder);

        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, (images, text): Self::Input<'_>) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        for image in images.iter() {
            let y = self.generate_one(engines, image, text)?;
            ys.push(Y::default().with_texts(&[y.into()]));
        }
        Ok(ys)
    }

    fn encode_images(&mut self, engines: &mut Engines, images: &[Image]) -> Result<Y> {
        let xs = self.image_processor.process(images)?;
        self.batch = images.len();
        let output = engines.run(&Module::Visual, inputs![xs]?)?;
        let x = output
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get vision output"))?;
        Ok(Y::default().with_embedding(X::from(x)))
    }
}
