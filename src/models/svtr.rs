use crate::{ops, DynConf, MinOptMax, Options, OrtEngine};
use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array, Axis, IxDyn};

#[derive(Debug)]
pub struct SVTR {
    engine: OrtEngine,
    pub height: MinOptMax,
    pub width: MinOptMax,
    pub batch: MinOptMax,
    confs: DynConf,
    vocab: Vec<String>,
}

impl SVTR {
    pub fn new(options: &Options) -> Result<Self> {
        let engine = OrtEngine::new(options)?;
        let (batch, height, width) = (
            engine.batch().to_owned(),
            engine.height().to_owned(),
            engine.width().to_owned(),
        );
        let confs = DynConf::new(&options.confs, 1);
        let mut vocab: Vec<_> =
            std::fs::read_to_string(options.vocab.as_ref().expect("No vocabulary found"))?
                .lines()
                .map(|line| line.to_string())
                .collect();
        vocab.push(" ".to_string());
        vocab.insert(0, "Blank".to_string());
        engine.dry_run()?;

        Ok(Self {
            engine,
            height,
            width,
            batch,
            vocab,
            confs,
        })
    }

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<Vec<String>> {
        let xs_ =
            ops::resize_with_fixed_height(xs, self.height.opt as u32, self.width.opt as u32, 0.0)?;
        let xs_ = ops::normalize(xs_, 0.0, 255.0);
        let ys: Vec<Array<f32, IxDyn>> = self.engine.run(&[xs_])?;
        let ys = ys[0].to_owned();

        self.postprocess(&ys)
    }

    pub fn postprocess(&self, output: &Array<f32, IxDyn>) -> Result<Vec<String>> {
        let mut texts: Vec<String> = Vec::new();
        for batch in output.axis_iter(Axis(0)) {
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

                    if idx == 0 || idx == self.vocab.len() - 1 {
                        return text_ids;
                    }

                    if *text_id != preds[idx - 1].0 {
                        text_ids.push(*text_id);
                    }
                    text_ids
                })
                .into_iter()
                .map(|idx| self.vocab[idx].to_owned())
                .collect::<String>();

            texts.push(text);
        }

        Ok(texts)
    }
}
