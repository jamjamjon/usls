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

    pub fn run(&mut self, xs: &[DynamicImage]) -> Result<()> {
        let xs_ =
            ops::resize_with_fixed_height(xs, self.height.opt as u32, self.width.opt as u32, 0.0)?;
        let xs_ = ops::normalize(xs_, 0.0, 255.0);
        let ys: Vec<Array<f32, IxDyn>> = self.engine.run(&[xs_])?;
        let ys = ys[0].to_owned();
        self.postprocess(&ys)?;
        Ok(())
    }

    pub fn postprocess(&self, xs: &Array<f32, IxDyn>) -> Result<()> {
        for batch in xs.axis_iter(Axis(0)) {
            let mut texts: Vec<String> = Vec::new();
            for (i, seq) in batch.axis_iter(Axis(0)).enumerate() {
                let (id, &confidence) = seq
                    .into_iter()
                    .enumerate()
                    .reduce(|max, x| if x.1 > max.1 { x } else { max })
                    .unwrap();
                if id == 0 || confidence < self.confs[0] {
                    continue;
                }
                if i == 0 && id == self.vocab.len() - 1 {
                    continue;
                }
                texts.push(self.vocab[id].to_owned());
            }
            texts.dedup();

            print!("[Texts] ");
            if texts.is_empty() {
                println!("Nothing detected!");
            } else {
                for text in texts.into_iter() {
                    print!("{text}");
                }
                println!();
            }
        }

        Ok(())
    }
}
