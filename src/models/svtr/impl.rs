use aksr::Builder;
use anyhow::Result;
use image::DynamicImage;
use ndarray::Axis;

use crate::{elapsed, DynConf, Engine, Options, Processor, Ts, Xs, Ys, Y};

#[derive(Builder, Debug)]
pub struct SVTR {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    confs: DynConf,
    spec: String,
    ts: Ts,
    processor: Processor,
}

impl SVTR {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let (batch, height, width, ts) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&960.into()).opt(),
            engine.try_width().unwrap_or(&960.into()).opt(),
            engine.ts.clone(),
        );
        let spec = options.model_spec().to_string();
        let confs = DynConf::new(options.class_confs(), 1);
        let processor = options
            .to_processor()?
            .with_image_width(width as _)
            .with_image_height(height as _);
        if processor.vocab().is_empty() {
            anyhow::bail!("No vocab file found")
        }

        Ok(Self {
            engine,
            height,
            width,
            batch,
            confs,
            processor,
            spec,
            ts,
        })
    }

    fn preprocess(&mut self, xs: &[DynamicImage]) -> Result<Xs> {
        Ok(self.processor.process_images(xs)?.into())
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[DynamicImage]) -> Result<Ys> {
        let ys = elapsed!("preprocess", self.ts, { self.preprocess(xs)? });
        let ys = elapsed!("inference", self.ts, { self.inference(ys)? });
        let ys = elapsed!("postprocess", self.ts, { self.postprocess(ys)? });

        Ok(ys)
    }

    pub fn summary(&mut self) {
        self.ts.summary();
    }

    pub fn postprocess(&self, xs: Xs) -> Result<Ys> {
        let mut ys: Vec<Y> = Vec::new();
        for batch in xs[0].axis_iter(Axis(0)) {
            let preds = batch
                .axis_iter(Axis(0))
                .filter_map(|x| {
                    x.into_iter()
                        .enumerate()
                        .max_by(|(_, x), (_, y)| x.total_cmp(y))
                })
                .collect::<Vec<_>>();

            let text = preds
                .iter()
                .enumerate()
                .fold(Vec::new(), |mut text_ids, (idx, (text_id, &confidence))| {
                    if *text_id == 0 || confidence < self.confs[0] {
                        return text_ids;
                    }

                    if idx == 0 || idx == self.processor.vocab().len() - 1 {
                        return text_ids;
                    }

                    if *text_id != preds[idx - 1].0 {
                        text_ids.push(*text_id);
                    }
                    text_ids
                })
                .into_iter()
                .map(|idx| self.processor.vocab()[idx].to_owned())
                .collect::<String>();

            ys.push(Y::default().with_texts(&[text.into()]))
        }

        Ok(ys.into())
    }
}
