use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Axis, IxDyn};
use std::io::Write;
use tokenizers::Tokenizer;

use crate::{ops, Embedding, LogitsSampler, MinOptMax, Options, OrtEngine, TokenizerStream, Y};

#[derive(Debug)]
pub struct Blip {
    pub textual: OrtEngine,
    pub visual: OrtEngine,
    pub height: MinOptMax,
    pub width: MinOptMax,
    pub batch_visual: MinOptMax,
    pub batch_textual: MinOptMax,
    tokenizer: TokenizerStream,
}

impl Blip {
    pub fn new(options_visual: Options, options_textual: Options) -> Result<Self> {
        let visual = OrtEngine::new(&options_visual)?;
        let textual = OrtEngine::new(&options_textual)?;
        let (batch_visual, batch_textual, height, width) = (
            visual.batch().to_owned(),
            textual.batch().to_owned(),
            visual.height().to_owned(),
            visual.width().to_owned(),
        );
        let tokenizer = Tokenizer::from_file(&options_textual.tokenizer.unwrap()).unwrap();
        let tokenizer = TokenizerStream::new(tokenizer);
        visual.dry_run()?;
        textual.dry_run()?;
        Ok(Self {
            textual,
            visual,
            batch_visual,
            batch_textual,
            height,
            width,
            tokenizer,
        })
    }

    pub fn encode_images(&self, xs: &[DynamicImage]) -> Result<Embedding> {
        let xs_ = ops::resize(xs, self.height.opt as u32, self.width.opt as u32)?;
        let xs_ = ops::normalize(xs_, 0.0, 255.0);
        let xs_ = ops::standardize(
            xs_,
            &[0.48145466, 0.4578275, 0.40821073],
            &[0.26862954, 0.2613026, 0.2757771],
        );
        let ys: Vec<Array<f32, IxDyn>> = self.visual.run(&[xs_])?;
        // let ys = ys[0].to_owned();
        Ok(Embedding::new(ys[0].to_owned()))
        // Ok(ys)
    }

    pub fn caption(
        &mut self,
        x: &[DynamicImage],
        prompt: Option<&str>,
        show: bool,
    ) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        let image_embeds = self.encode_images(x)?;
        let image_embeds_attn_mask: Array<f32, IxDyn> =
            Array::ones((1, image_embeds.embedding().shape()[1])).into_dyn();
        let mut y_text = String::new();

        // conditional
        let mut input_ids = match prompt {
            None => {
                if show {
                    print!("[Unconditional]: ");
                }
                vec![0.0f32]
            }
            Some(prompt) => {
                let encodings = self.tokenizer.tokenizer().encode(prompt, false);
                let ids: Vec<f32> = encodings
                    .unwrap()
                    .get_ids()
                    .iter()
                    .map(|x| *x as f32)
                    .collect();
                if show {
                    print!("[Conditional]: {} ", prompt);
                }
                y_text.push_str(&format!("{} ", prompt));
                ids
            }
        };

        let mut logits_sampler = LogitsSampler::new();
        loop {
            let input_ids_nd: Array<f32, IxDyn> = Array::from_vec(input_ids.to_owned()).into_dyn();
            let input_ids_nd = input_ids_nd.insert_axis(Axis(0));
            let input_ids_attn_mask: Array<f32, IxDyn> =
                Array::ones(input_ids_nd.shape()).into_dyn();
            let y = self.textual.run(&[
                input_ids_nd,
                input_ids_attn_mask,
                image_embeds.embedding().to_owned(),
                image_embeds_attn_mask.to_owned(),
            ])?; // N, length, vocab_size
            let y = y[0].slice(s!(0, -1.., ..));
            let logits = y.slice(s!(0, ..)).to_vec();
            let token_id = logits_sampler.decode(&logits)?;
            input_ids.push(token_id as f32);

            // SEP
            if token_id == 102 {
                break;
            }

            // streaming generation
            if let Some(t) = self.tokenizer.next_token(token_id as u32)? {
                y_text.push_str(&t);
                if show {
                    print!("{t}");
                    // std::thread::sleep(std::time::Duration::from_millis(5));
                }
                std::io::stdout().flush()?;
            }
        }
        if show {
            println!();
        }
        self.tokenizer.clear();
        ys.push(Y::default().with_texts(&[y_text]));
        Ok(ys)
    }

    pub fn batch_visual(&self) -> usize {
        self.batch_visual.opt as usize
    }

    pub fn batch_textual(&self) -> usize {
        self.batch_textual.opt as usize
    }
}
