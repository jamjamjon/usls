use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Axis, IxDyn};
use std::io::Write;
use tokenizers::Tokenizer;

use crate::{auto_load, ops, MinOptMax, Options, OrtEngine, TokenizerStream};

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
        let tokenizer = match &options_textual.tokenizer {
            None => auto_load("tokenizer-blip.json")?,
            Some(tokenizer) => tokenizer.into(),
        };
        let tokenizer = Tokenizer::from_file(tokenizer).unwrap();
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

    pub fn encode_images(&self, xs: &[DynamicImage]) -> Result<Array<f32, IxDyn>> {
        let xs_ = ops::resize(xs, self.height.opt as u32, self.width.opt as u32, true)?;
        let ys: Vec<Array<f32, IxDyn>> = self.visual.run(&[xs_])?;
        let ys = ys[0].to_owned();
        Ok(ys)
    }

    pub fn caption(&mut self, path: &str, prompt: Option<&str>) -> Result<()> {
        // this demo use batch_size=1
        let x = image::io::Reader::open(path)?.decode()?;
        let image_embeds = self.encode_images(&[x])?;
        let image_embeds_attn_mask: Array<f32, IxDyn> =
            Array::ones((1, image_embeds.shape()[1])).into_dyn();

        // conditional
        let mut input_ids = match prompt {
            None => {
                print!("[Unconditional image captioning]: ");
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
                print!("[Conditional image captioning]: {} ", prompt);
                ids
            }
        };
        loop {
            let input_ids_nd: Array<f32, IxDyn> = Array::from_vec(input_ids.to_owned()).into_dyn();
            let input_ids_nd = input_ids_nd.insert_axis(Axis(0));
            let input_ids_attn_mask: Array<f32, IxDyn> =
                Array::ones(input_ids_nd.shape()).into_dyn();
            let y = self.textual.run(&[
                input_ids_nd,
                input_ids_attn_mask,
                image_embeds.to_owned(),
                image_embeds_attn_mask.to_owned(),
            ])?; // N, length, vocab_size
            let y = y[0].to_owned();
            let y = y.slice(s!(0, -1.., ..));

            // softmax
            let exps = y.mapv(|c| c.exp());
            let stds = exps.sum_axis(Axis(1));
            let probs = exps / stds.insert_axis(Axis(1));
            let probs = probs.slice(s!(0, ..));

            // argmax
            let (token_id, _) = probs
                .into_iter()
                .enumerate()
                .reduce(|max, x| if x.1 > max.1 { x } else { max })
                .unwrap();
            input_ids.push(token_id as f32);

            // SEP
            if token_id == 102 {
                break;
            }

            // streaming generation
            if let Some(t) = self.tokenizer.next_token(token_id as u32)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            // sleep for test
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        println!();
        self.tokenizer.clear();
        Ok(())
    }

    pub fn batch_visual(&self) -> usize {
        self.batch_visual.opt as usize
    }

    pub fn batch_textual(&self) -> usize {
        self.batch_textual.opt as usize
    }
}
