use aksr::Builder;
use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Axis};

use crate::{
    elapsed,
    models::{BaseModelTextual, BaseModelVisual},
    LogitsSampler, Options, Ts, Xs, Ys, X, Y,
};

#[derive(Debug, Builder)]
pub struct Blip {
    visual: BaseModelVisual,
    textual: BaseModelTextual,
    ts: Ts,
    max_length: usize,
    eos_token_id: u32,
}

impl Blip {
    pub fn new(options_visual: Options, options_textual: Options) -> Result<Self> {
        let visual = BaseModelVisual::new(options_visual)?;
        let textual = BaseModelTextual::new(options_textual)?;
        let ts = Ts::merge(&[visual.engine().ts(), textual.engine().ts()]);
        let max_length = 512;
        let eos_token_id = 102;

        Ok(Self {
            textual,
            visual,
            ts,
            max_length,
            eos_token_id,
        })
    }

    pub fn encode_images(&mut self, xs: &[DynamicImage]) -> Result<X> {
        self.visual.encode(xs)
    }

    pub fn encode_texts(&mut self, text: Option<&str>) -> Result<Vec<Vec<f32>>> {
        let input_ids = self
            .textual
            .processor()
            .encode_text_ids(text.unwrap_or_default(), false)?;
        Ok(vec![input_ids.clone(); self.batch()])
    }

    pub fn forward(&mut self, images: &[DynamicImage], text: Option<&str>) -> Result<Ys> {
        let image_embeds = elapsed!("encode_images", self.ts, { self.encode_images(images)? });
        let ys = elapsed!("generate", self.ts, { self.generate(&image_embeds, text)? });

        Ok(ys)
    }

    pub fn generate(&mut self, image_embeds: &X, text: Option<&str>) -> Result<Ys> {
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
            let outputs = self.textual.inference(Xs::from(vec![
                input_ids_nd,
                input_ids_attn_mask,
                image_embeds.clone(),
                X::ones(&[self.visual().batch(), image_embeds.dims()[1]]), // image_embeds_attn_mask
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
        let texts = self.textual.processor().decode_tokens_batch(
            &token_ids
                .into_iter()
                .map(|v| v.into_iter().map(|x| x as u32).collect::<Vec<_>>())
                .collect::<Vec<Vec<_>>>(),
            true,
        )?;

        let ys = texts
            .into_iter()
            .map(|x| Y::default().with_texts(&[x.into()]))
            .collect::<Vec<_>>()
            .into();

        Ok(ys)
    }

    pub fn summary(&mut self) {
        self.ts.summary();
    }

    pub fn batch(&self) -> usize {
        self.visual.batch() as _
    }
}
