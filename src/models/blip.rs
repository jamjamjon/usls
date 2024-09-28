use anyhow::Result;
use image::DynamicImage;
use ndarray::s;
use std::io::Write;
use tokenizers::Tokenizer;

use crate::{
    Embedding, LogitsSampler, MinOptMax, Ops, Options, OrtEngine, TokenizerStream, Xs, X, Y,
};

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
        let mut visual = OrtEngine::new(&options_visual)?;
        let mut textual = OrtEngine::new(&options_textual)?;
        let (batch_visual, batch_textual, height, width) = (
            visual.batch().to_owned(),
            textual.batch().to_owned(),
            visual.height().to_owned(),
            visual.width().to_owned(),
        );

        let tokenizer = options_textual
            .tokenizer
            .ok_or(anyhow::anyhow!("No tokenizer file found"))?;
        let tokenizer = match Tokenizer::from_file(tokenizer) {
            Err(err) => anyhow::bail!("Failed to build tokenizer: {:?}", err),
            Ok(x) => x,
        };

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

    pub fn encode_images(&mut self, xs: &[DynamicImage]) -> Result<Y> {
        let xs_ = X::apply(&[
            Ops::Resize(
                xs,
                self.height.opt() as u32,
                self.width.opt() as u32,
                "Bilinear",
            ),
            Ops::Normalize(0., 255.),
            Ops::Standardize(
                &[0.48145466, 0.4578275, 0.40821073],
                &[0.26862954, 0.2613026, 0.2757771],
                3,
            ),
            Ops::Nhwc2nchw,
        ])?;
        let ys = self.visual.run(Xs::from(xs_))?;
        Ok(Y::default().with_embedding(&Embedding::from(ys[0].to_owned())))
    }

    pub fn caption(&mut self, xs: &Y, prompt: Option<&str>, show: bool) -> Result<Vec<Y>> {
        let mut ys: Vec<Y> = Vec::new();
        let image_embeds = match xs.embedding() {
            Some(x) => X::from(x.data().to_owned()),
            None => anyhow::bail!("No image embeddings found."),
        };
        let image_embeds_attn_mask = X::ones(&[self.batch_visual(), image_embeds.dims()[1]]);

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
                let encodings = match self.tokenizer.tokenizer().encode(prompt, false) {
                    Err(err) => anyhow::bail!("{}", err),
                    Ok(x) => x,
                };
                let ids: Vec<f32> = encodings.get_ids().iter().map(|x| *x as f32).collect();
                if show {
                    print!("[Conditional]: {} ", prompt);
                }
                y_text.push_str(&format!("{} ", prompt));
                ids
            }
        };

        let mut logits_sampler = LogitsSampler::new();
        loop {
            let input_ids_nd = X::from(input_ids.to_owned())
                .insert_axis(0)?
                .repeat(0, self.batch_textual())?;
            let input_ids_attn_mask = X::ones(input_ids_nd.dims());

            let y = self.textual.run(Xs::from(vec![
                input_ids_nd,
                input_ids_attn_mask,
                image_embeds.clone(),
                image_embeds_attn_mask.clone(),
            ]))?; // N, length, vocab_size
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
        self.batch_visual.opt()
    }

    pub fn batch_textual(&self) -> usize {
        self.batch_textual.opt()
    }
}
