use crate::{ops, Embedding, MinOptMax, Options, OrtEngine};
use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array, Array2, IxDyn};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

#[derive(Debug)]
pub struct Clip {
    pub textual: OrtEngine,
    pub visual: OrtEngine,
    pub height: MinOptMax,
    pub width: MinOptMax,
    pub batch_visual: MinOptMax,
    pub batch_textual: MinOptMax,
    tokenizer: Tokenizer,
    context_length: usize,
}

impl Clip {
    pub fn new(options_visual: Options, options_textual: Options) -> Result<Self> {
        let context_length = 77;
        let mut visual = OrtEngine::new(&options_visual)?;
        let mut textual = OrtEngine::new(&options_textual)?;
        let (batch_visual, batch_textual, height, width) = (
            visual.inputs_minoptmax()[0][0].to_owned(),
            textual.inputs_minoptmax()[0][0].to_owned(),
            visual.inputs_minoptmax()[0][2].to_owned(),
            visual.inputs_minoptmax()[0][3].to_owned(),
        );
        let mut tokenizer = Tokenizer::from_file(&options_textual.tokenizer.unwrap()).unwrap();
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(context_length),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));

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
            context_length,
        })
    }

    pub fn encode_images(&mut self, xs: &[DynamicImage]) -> Result<Embedding> {
        let xs_ = ops::resize(xs, self.height.opt as u32, self.width.opt as u32)?;
        let xs_ = ops::normalize(xs_, 0.0, 255.0);
        let xs_ = ops::standardize(
            xs_,
            &[0.48145466, 0.4578275, 0.40821073],
            &[0.26862954, 0.2613026, 0.2757771],
        );
        let ys: Vec<Array<f32, IxDyn>> = self.visual.run(&[xs_])?;
        Ok(Embedding::new(ys[0].to_owned()))
    }

    pub fn encode_texts(&mut self, texts: &[String]) -> Result<Embedding> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_owned(), false)
            .unwrap();
        let xs: Vec<f32> = encodings
            .iter()
            .flat_map(|i| i.get_ids().iter().map(|&b| b as f32))
            .collect();
        let xs = Array2::from_shape_vec((texts.len(), self.context_length), xs)?.into_dyn();
        let ys = self.textual.run(&[xs])?;
        Ok(Embedding::new(ys[0].to_owned()))
    }

    pub fn batch_visual(&self) -> usize {
        self.batch_visual.opt as usize
    }

    pub fn batch_textual(&self) -> usize {
        self.batch_textual.opt as usize
    }
}
