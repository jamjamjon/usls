use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{elapsed_module, Config, DynConf, Engine, Hbb, Image, Processor, Xs, X, Y};

#[derive(Debug, Builder)]
pub struct PicoDet {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    spec: String,
    names: Vec<String>,
    confs: DynConf,
    processor: Processor,
}

impl PicoDet {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&640.into()).opt(),
            engine.try_width().unwrap_or(&640.into()).opt(),
        );
        let spec = engine.spec().to_owned();
        let names: Vec<String> = config.class_names().to_vec();
        let confs = DynConf::new_or_default(config.class_confs(), names.len());
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            engine,
            height,
            width,
            batch,
            spec,
            names,
            confs,
            processor,
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        let x1 = self.processor.process_images(xs)?;
        let x2: X = self
            .processor
            .images_transform_info
            .iter()
            .map(|x| vec![x.height_scale, x.width_scale])
            .collect::<Vec<_>>()
            .try_into()?;

        Ok(Xs::from(vec![x1, x2]))
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("PicoDet", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("PicoDet", "inference", self.inference(ys)?);
        let ys = elapsed_module!("PicoDet", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }
    fn postprocess(&mut self, xs: Xs) -> Result<Vec<Y>> {
        // ONNX models exported by paddle2onnx
        // TODO: ONNX model's batch size seems always = 1
        // xs[0] : n, 6
        // xs[1] : n
        let y_bboxes: Vec<Hbb> = xs[0]
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(_i, pred)| {
                let (class_id, confidence) = (pred[0] as usize, pred[1]);
                if confidence < self.confs[class_id] {
                    return None;
                }
                let (x1, y1, x2, y2) = (pred[2], pred[3], pred[4], pred[5]);
                let mut hbb = Hbb::default()
                    .with_xyxy(x1.max(0.0f32), y1.max(0.0f32), x2, y2)
                    .with_confidence(confidence)
                    .with_id(class_id);
                if !self.names.is_empty() {
                    hbb = hbb.with_name(&self.names[class_id]);
                }

                Some(hbb)
            })
            .collect();

        let mut y = Y::default();
        if !y_bboxes.is_empty() {
            y = y.with_hbbs(&y_bboxes);
        }

        Ok(vec![y])
    }
}
