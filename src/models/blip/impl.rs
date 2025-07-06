use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};

use crate::{elapsed_module, Config, Engine, Image, LogitsSampler, Processor, Xs, X, Y};

#[derive(Debug, Builder)]
pub struct Blip {
    visual: Engine,
    textual: Engine,
    batch: usize,
    height: usize,
    width: usize,
    processor: Processor,
    max_length: usize,
    eos_token_id: u32,
}

impl Blip {
    pub fn new(config: Config) -> Result<Self> {
        let visual = Engine::try_from_config(&config.visual)?;
        let textual = Engine::try_from_config(&config.textual)?;
        let (batch, height, width) = (
            visual.batch().opt(),
            visual.try_height().unwrap_or(&384.into()).opt(),
            visual.try_width().unwrap_or(&384.into()).opt(),
        );

        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let max_length = 512;
        let eos_token_id = 102;

        Ok(Self {
            textual,
            visual,
            max_length,
            eos_token_id,
            batch,
            height,
            width,
            processor,
        })
    }

    pub fn encode_images(&mut self, xs: &[Image]) -> Result<X> {
        let ys = self.processor.process_images(xs)?;
        self.batch = xs.len(); // update
        let ys = self.visual.run(ys.into())?;

        Ok(ys[0].to_owned())
    }

    pub fn encode_texts(&mut self, text: Option<&str>) -> Result<Vec<Vec<f32>>> {
        let input_ids = self
            .processor
            .encode_text_ids(text.unwrap_or_default(), false)?;
        Ok(vec![input_ids.clone(); self.batch()])
    }

    pub fn forward(&mut self, images: &[Image], text: Option<&str>) -> Result<Vec<Y>> {
        let image_embeds = elapsed_module!("BLIP", "encode_images", self.encode_images(images)?);
        let ys = elapsed_module!("BLIP", "generate", self.generate(&image_embeds, text)?);

        Ok(ys)
    }

    pub fn generate(&mut self, image_embeds: &X, text: Option<&str>) -> Result<Vec<Y>> {
        // encode texts
        let mut token_ids = self.encode_texts(text)?;

        // generate
        let logits_sampler = LogitsSampler::new();
        let mut finished = vec![false; self.batch()];
        for _ in 0..self.max_length {
            let input_ids_nd = token_ids
                .iter()
                .map(|tokens| X::from(tokens.clone()).insert_axis(0))
                .collect::<Result<Vec<_>, _>>()?;

            let input_ids_nd = X::concat(&input_ids_nd, 0)?;
            let input_ids_attn_mask = X::ones(input_ids_nd.dims());

            // decode
            let outputs = self.textual.run(Xs::from(vec![
                input_ids_nd,
                input_ids_attn_mask,
                image_embeds.clone(),
                X::ones(&[self.batch(), image_embeds.dims()[1]]),
            ]))?;

            // decode each token for each batch
            for (i, logit) in outputs[0].axis_iter(Axis(0)).enumerate() {
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
                    }
                    token_ids[i].push(token_id as f32);
                } else {
                    token_ids[i].push(self.eos_token_id as f32);
                }
            }

            if finished.iter().all(|&x| x) {
                break;
            }
        }

        // batch decode
        let texts = self.processor.decode_tokens_batch(
            &token_ids
                .into_iter()
                .map(|v| v.into_iter().map(|x| x as u32).collect::<Vec<_>>())
                .collect::<Vec<Vec<_>>>(),
            true,
        )?;

        let ys = texts
            .into_iter()
            .map(|x| Y::default().with_texts(&[x.into()]))
            .collect::<Vec<_>>();

        Ok(ys)
    }
}
