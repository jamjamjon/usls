use anyhow::Result;
use image::DynamicImage;
use ndarray::s;
use std::io::Write;
use tokenizers::Tokenizer;

use crate::{LogitsSampler, MinOptMax, Ops, Options, OrtEngine, Task, TokenizerStream, Xs, X, Y};

#[derive(Debug)]
pub struct Florence2 {
    pub vision_encoder: OrtEngine,
    pub text_embed: OrtEngine,
    pub encoder: OrtEngine,
    pub decoder: OrtEngine,
    pub decoder_merged: OrtEngine,
    pub height: MinOptMax,
    pub width: MinOptMax,
    pub batch: MinOptMax,
    tokenizer: TokenizerStream,
    task: Task,
}

impl Florence2 {
    pub fn new(
        options_vision_encoder: Options,
        options_text_embed: Options,
        options_encoder: Options,
        options_decoder: Options,
        options_decoder_merged: Options,
    ) -> Result<Self> {
        let mut vision_encoder = OrtEngine::new(&options_vision_encoder)?;
        let mut text_embed = OrtEngine::new(&options_text_embed)?;
        let mut encoder = OrtEngine::new(&options_encoder)?;
        let mut decoder = OrtEngine::new(&options_decoder)?;
        let mut decoder_merged = OrtEngine::new(&options_decoder_merged)?;
        let (batch, height, width) = (
            vision_encoder.batch().to_owned(),
            vision_encoder.height().to_owned(),
            vision_encoder.width().to_owned(),
        );
        let task = options_text_embed.task;
        let tokenizer = options_text_embed
            .tokenizer
            .ok_or(anyhow::anyhow!("No tokenizer file found"))?;
        let tokenizer = match Tokenizer::from_file(tokenizer) {
            Err(err) => anyhow::bail!("Failed to build tokenizer: {:?}", err),
            Ok(x) => x,
        };

        let tokenizer = TokenizerStream::new(tokenizer);

        // dry run
        vision_encoder.dry_run()?;
        text_embed.dry_run()?;
        encoder.dry_run()?;
        decoder.dry_run()?;
        decoder_merged.dry_run()?;

        Ok(Self {
            vision_encoder,
            text_embed,
            encoder,
            decoder,
            decoder_merged,
            height,
            width,
            batch,
            tokenizer,
            task,
        })
    }

    pub fn encode_images(&mut self, xs: &[DynamicImage]) -> Result<X> {
        let xs_ = X::apply(&[
            Ops::Resize(
                xs,
                self.height.opt as u32,
                self.width.opt as u32,
                "Bilinear",
            ),
            Ops::Normalize(0., 255.),
            Ops::Standardize(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225], 3),
            Ops::Nhwc2nchw,
        ])?;
        let ys = self.vision_encoder.run(Xs::from(xs_))?[0].to_owned();
        Ok(ys)
    }

    pub fn caption(&mut self, image_embeddings: &X, display: bool) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();

        // encode prompt
        let input_ids = self
            .construct_prompt(None)?
            .insert_axis(0)?
            .repeat(0, self.batch())?;
        let text_embedings = self.text_embed.run(Xs::from(input_ids))?[0]
            .clone()
            .repeat(0, self.batch())?;

        // concate image_embeddings and prompt embeddings
        let inputs_embeds = image_embeddings.clone().concatenate(&text_embedings, 1)?;

        let attention_mask = X::ones(&[self.batch(), inputs_embeds.dims()[1]]);

        // encoder
        let last_hidden_state = self.encoder.run(Xs::from(vec![
            attention_mask.clone(),
            inputs_embeds.clone(),
        ]))?[0]
            .clone();

        // decoder
        let inputs_embeds = inputs_embeds.slice(s![.., -1.., ..]);
        let inputs_embeds = X::from(inputs_embeds.to_owned().into_dyn());
        let mut decoder_outputs = self.decoder.run(Xs::from(vec![
            attention_mask.clone(),
            last_hidden_state.clone(),
            inputs_embeds,
        ]))?;

        let encoder_k0 = decoder_outputs[3].clone();
        let encoder_v0 = decoder_outputs[4].clone();

        let encoder_k1 = decoder_outputs[7].clone();
        let encoder_v1 = decoder_outputs[8].clone();

        let encoder_k2 = decoder_outputs[11].clone();
        let encoder_v2 = decoder_outputs[12].clone();

        let encoder_k3 = decoder_outputs[15].clone();
        let encoder_v3 = decoder_outputs[16].clone();

        let encoder_k4 = decoder_outputs[19].clone();
        let encoder_v4 = decoder_outputs[20].clone();

        let encoder_k5 = decoder_outputs[23].clone();
        let encoder_v5 = decoder_outputs[24].clone();

        let mut y_text = String::new();
        let mut generated_tokens = Vec::new();

        // TODO: batch iter
        let mut logits_sampler = LogitsSampler::new();
        loop {
            let logits = &decoder_outputs["logits"];
            let decoder_k0 = &decoder_outputs[1];
            let decoder_v0 = &decoder_outputs[2];
            let decoder_k1 = &decoder_outputs[5];
            let decoder_v1 = &decoder_outputs[6];
            let decoder_k2 = &decoder_outputs[9];
            let decoder_v2 = &decoder_outputs[10];
            let decoder_k3 = &decoder_outputs[13];
            let decoder_v3 = &decoder_outputs[14];
            let decoder_k4 = &decoder_outputs[17];
            let decoder_v4 = &decoder_outputs[18];
            let decoder_k5 = &decoder_outputs[21];
            let decoder_v5 = &decoder_outputs[22];

            let next_token_logits = logits
                .slice(s![.., -1.., ..])
                .to_owned()
                .into_raw_vec_and_offset()
                .0;

            let token_id = logits_sampler.decode(&next_token_logits)?;
            generated_tokens.push(token_id as f32);

            // </s>
            if token_id == 2 {
                break;
            }

            // streaming generation
            if let Some(t) = self.tokenizer.next_token(token_id)? {
                y_text.push_str(&t);
                if display {
                    print!("{t}");
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                std::io::stdout().flush()?;
            }

            // next input text embedding
            let next_token = X::from(vec![token_id as f32])
                .insert_axis(0)?
                .repeat(0, self.batch())?;

            // decode
            let inputs_embeds = &self.text_embed.run(Xs::from(next_token))?[0].clone();
            let use_cache = X::ones(&[1]);
            decoder_outputs = self.decoder_merged.run(Xs::from(vec![
                attention_mask.clone(),
                last_hidden_state.clone(),
                inputs_embeds.clone(),
                decoder_k0.clone(),
                decoder_v0.clone(),
                encoder_k0.clone(),
                encoder_v0.clone(),
                decoder_k1.clone(),
                decoder_v1.clone(),
                encoder_k1.clone(),
                encoder_v1.clone(),
                decoder_k2.clone(),
                decoder_v2.clone(),
                encoder_k2.clone(),
                encoder_v2.clone(),
                decoder_k3.clone(),
                decoder_v3.clone(),
                encoder_k3.clone(),
                encoder_v3.clone(),
                decoder_k4.clone(),
                decoder_v4.clone(),
                encoder_k4.clone(),
                encoder_v4.clone(),
                decoder_k5.clone(),
                decoder_v5.clone(),
                encoder_k5.clone(),
                encoder_v5.clone(),
                use_cache,
            ]))?;
        }
        if display {
            println!();
        }
        self.tokenizer.clear();

        ys.push(Y::default().with_texts(&[y_text]));

        Ok(ys)
    }

    pub fn construct_prompt(&self, text: Option<&str>) -> Result<X> {
        let prompt = match self.task {
            Task::Untitled => anyhow::bail!("No task specified."),
            Task::Caption(0) => "What does the image describe?".to_string(),
            Task::Caption(1) => "Describe in detail what is shown in the image.".to_string(),
            Task::Caption(2) => "Describe with a paragraph what is shown in the image.".to_string(),
            Task::Ocr => "What is the text in the image?".to_string(),
            Task::OcrWithRegion => "What is the text in the image, with regions?".to_string(),
            Task::ObjectDetection => {
                "Locate the objects with category name in the image.".to_string()
            }
            Task::DenseRegionCaption => {
                "Locate the objects in the image, with their descriptions.".to_string()
            }
            Task::RegionProposal => "Locate the region proposals in the image.".to_string(),
            Task::PhraseGrounding => format!(
                "Locate the phrases in the caption: {}",
                text.unwrap_or_default()
            ),
            Task::ReferringExpressionSegmentation => {
                format!("Locate {} in the image with mask", text.unwrap_or_default())
            }
            Task::RegionToSegmentation => {
                format!(
                    "What is the polygon mask of region {}",
                    text.unwrap_or_default()
                )
            }
            Task::OpenSetDetection => {
                format!("Locate {} in the image.", text.unwrap_or_default())
            }
            Task::RegionToCategory => {
                format!("What is the region {}?", text.unwrap_or_default())
            }
            Task::RegionToDescription => {
                format!(
                    "What does the region {} describe?",
                    text.unwrap_or_default()
                )
            }
            Task::RegionToOcr => {
                format!("What text is in the region {}?", text.unwrap_or_default())
            }

            _ => todo!(),
        };

        let encodings = match self.tokenizer.tokenizer().encode(prompt, true) {
            Err(err) => anyhow::bail!("{}", err),
            Ok(x) => x,
        };
        let ids: Vec<f32> = encodings.get_ids().iter().map(|x| *x as f32).collect();

        let ids = X::from(ids);
        Ok(ids)
    }

    pub fn with_task(mut self, task: Task) -> Self {
        self.task = task;
        self
    }

    pub fn batch(&self) -> usize {
        self.batch.opt as usize
    }
}
