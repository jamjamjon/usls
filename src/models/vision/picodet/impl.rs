use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{
    inputs, Config, DynConf, Engine, Engines, FromConfig, Hbb, Image, ImageProcessor, Model,
    Module, Xs, X, Y,
};

/// PP-PicoDet: A Better Real-Time Object Detector
#[derive(Debug, Builder)]
pub struct PicoDet {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub spec: String,
    pub names: Vec<String>,
    pub confs: DynConf,
    pub processor: ImageProcessor,
}

impl Model for PicoDet {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&640.into()).opt(),
            engine.try_width().unwrap_or(&640.into()).opt(),
        );
        let spec = engine.spec().to_owned();
        let names: Vec<String> = config.inference.class_names;
        let confs = DynConf::new_or_default(&config.inference.class_confs, names.len());
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            height,
            width,
            batch,
            spec,
            names,
            confs,
            processor,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let x1 = crate::perf!("PicoDet::preprocess", self.processor.process(images)?);
        let x2: X = self
            .processor
            .images_transform_info
            .iter()
            .map(|x| vec![x.height_scale, x.width_scale])
            .collect::<Vec<_>>()
            .try_into()?;
        let ys = crate::perf!(
            "PicoDet::inference",
            engines.run(&Module::Model, inputs![&x1, x2]?)?
        );
        crate::perf!("PicoDet::postprocess", self.postprocess(&ys))
    }
}

impl PicoDet {
    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let xs = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output"))?;
        // ONNX models exported by paddle2onnx
        // TODO: ONNX model's batch size seems always = 1
        // xs[0] : n, 6
        // xs[1] : n
        let y_bboxes: Vec<Hbb> = xs
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

        Ok(vec![Y::default().with_hbbs(&y_bboxes)])
    }
}
