use aksr::Builder;
use anyhow::Result;
use ndarray::Array2;

use crate::{elapsed, Engine, Image, Options, Processor, Ts, Xs, X};

#[derive(Debug, Builder)]
pub struct ClipVisual {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    processor: Processor,
    ts: Ts,
}

impl ClipVisual {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let (batch, height, width, ts) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&224.into()).opt(),
            engine.try_width().unwrap_or(&224.into()).opt(),
            engine.ts.clone(),
        );
        let processor = options
            .to_processor()?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            engine,
            height,
            width,
            batch,
            processor,
            ts,
        })
    }

    pub fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        let x = self.processor.process_images(xs)?;

        Ok(x.into())
    }

    pub fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn encode_images(&mut self, xs: &[Image]) -> Result<X> {
        let xs = elapsed!("visual-preprocess", self.ts, { self.preprocess(xs)? });
        let xs = elapsed!("visual-inference", self.ts, { self.inference(xs)? });
        let x = elapsed!("visual-postprocess", self.ts, { xs[0].to_owned() });

        Ok(x)
    }
}

#[derive(Debug, Builder)]
pub struct ClipTextual {
    engine: Engine,
    batch: usize,
    processor: Processor,
    ts: Ts,
}

impl ClipTextual {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let (batch, ts) = (engine.batch().opt(), engine.ts.clone());
        let processor = options.to_processor()?;

        Ok(Self {
            engine,
            batch,
            processor,
            ts,
        })
    }

    pub fn preprocess(&self, xs: &[&str]) -> Result<Xs> {
        let encodings: Vec<f32> = self
            .processor
            .encode_texts_ids(xs, false)? // skip_special_tokens
            .into_iter()
            .flatten()
            .collect();

        let x: X = Array2::from_shape_vec((xs.len(), encodings.len() / xs.len()), encodings)?
            .into_dyn()
            .into();

        Ok(x.into())
    }

    pub fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn encode_texts(&mut self, xs: &[&str]) -> Result<X> {
        let xs = elapsed!("textual-preprocess", self.ts, { self.preprocess(xs)? });
        let xs = elapsed!("textual-inference", self.ts, { self.inference(xs)? });
        let x = elapsed!("textual-postprocess", self.ts, { xs[0].to_owned() });

        Ok(x)
    }
}

#[derive(Debug, Builder)]
pub struct Clip {
    textual: ClipTextual,
    visual: ClipVisual,
    ts: Ts,
}

impl Clip {
    pub fn new(options_visual: Options, options_textual: Options) -> Result<Self> {
        let visual = ClipVisual::new(options_visual)?;
        let textual = ClipTextual::new(options_textual)?;
        // let ts = Ts::merge(&[visual.engine().ts(), textual.engine().ts()]);
        let ts = Ts::default();

        Ok(Self {
            textual,
            visual,
            ts,
        })
    }

    pub fn encode_images(&mut self, xs: &[Image]) -> Result<X> {
        let x = elapsed!("encode_images", self.ts, { self.visual.encode_images(xs)? });
        Ok(x)
    }

    pub fn encode_texts(&mut self, xs: &[&str]) -> Result<X> {
        let x = elapsed!("encode_texts", self.ts, { self.textual.encode_texts(xs)? });
        Ok(x)
    }

    pub fn summary(&mut self) {
        // self.ts.clear();
        // self.ts = Ts::merge(&[&self.ts, self.visual.ts(), self.textual.ts()]);
        self.ts.summary();
        self.visual.ts().summary();
        self.textual.ts().summary();
    }
}
