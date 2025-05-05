use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{
    elapsed, impl_model_config_methods, impl_process_config_methods, DynConf, Engine, Image,
    Options, Prob, Processor, Ts, Xs, Y,
};

#[derive(Debug, Builder)]
pub struct ImageClassifier {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    apply_softmax: bool,
    ts: Ts,
    processor: Processor,
    confs: DynConf,
    nc: usize,
    names: Vec<String>,
    spec: String,
}

impl TryFrom<Options> for ImageClassifier {
    type Error = anyhow::Error;

    fn try_from(options: Options) -> Result<Self, Self::Error> {
        let engine = options.to_engine()?;
        let spec = engine.spec().to_string();
        let (batch, height, width, ts) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&224.into()).opt(),
            engine.try_width().unwrap_or(&224.into()).opt(),
            engine.ts().clone(),
        );
        let processor = options
            .to_processor()?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let (nc, names) = match (options.nc(), options.class_names()) {
            (Some(nc), Some(names)) => {
                if nc != names.len() {
                    anyhow::bail!(
                        "The length of the input class names: {} is inconsistent with the number of classes: {}.",
                        names.len(),
                        nc
                    );
                }
                (nc, names.to_vec())
            }
            (Some(nc), None) => (
                nc,
                (0..nc).map(|x| format!("# {}", x)).collect::<Vec<String>>(),
            ),
            (None, Some(names)) => (names.len(), names.to_vec()),
            (None, None) => {
                anyhow::bail!("Neither class names nor class numbers were specified.");
            }
        };
        let confs = DynConf::new(options.class_confs(), nc);
        let apply_softmax = options.apply_softmax.unwrap_or_default();

        Ok(Self {
            engine,
            height,
            width,
            batch,
            nc,
            ts,
            spec,
            processor,
            confs,
            names,
            apply_softmax,
        })
    }
}

impl ImageClassifier {
    pub fn summary(&mut self) {
        self.ts.summary();
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        let x = self.processor.process_images(xs)?;

        Ok(x.into())
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed!("preprocess", self.ts, { self.preprocess(xs)? });
        let ys = elapsed!("inference", self.ts, { self.inference(ys)? });
        let ys = elapsed!("postprocess", self.ts, { self.postprocess(ys)? });

        Ok(ys)
    }

    fn postprocess(&self, xs: Xs) -> Result<Vec<Y>> {
        let ys: Vec<Y> = xs[0]
            .axis_iter(Axis(0))
            .into_par_iter()
            .filter_map(|logits| {
                let logits = if self.apply_softmax {
                    let exps = logits.mapv(|x| x.exp());
                    let stds = exps.sum_axis(Axis(0));
                    exps / stds
                } else {
                    logits.into_owned()
                };
                let probs = Prob::new_probs(
                    &logits.into_raw_vec_and_offset().0,
                    Some(&self.names.iter().map(|x| x.as_str()).collect::<Vec<_>>()),
                    3,
                );

                Some(Y::default().with_probs(&probs))
            })
            .collect::<Vec<_>>();

        Ok(ys)
    }
}

#[derive(aksr::Builder, Debug, Clone)]
pub struct ImageClassificationConfig {
    pub model: crate::ModelConfig,
    pub processor: crate::ProcessorConfig,
    pub class_confs: Vec<f32>,
    pub class_names: Vec<String>,
    pub apply_softmax: bool,
    pub topk: usize,
}

impl Default for ImageClassificationConfig {
    fn default() -> Self {
        Self {
            model: crate::ModelConfig::default()
                .with_ixx(0, 0, 1.into())
                .with_ixx(0, 1, 3.into())
                .with_ixx(0, 2, 224.into())
                .with_ixx(0, 3, 224.into()),
            processor: crate::ProcessorConfig::default()
                .with_resize_mode(crate::ResizeMode::FitExact)
                .with_normalize(true)
                .with_image_mean(&[0.48145466, 0.4578275, 0.40821073])
                .with_image_std(&[0.26862954, 0.2613026, 0.2757771]),
            class_confs: vec![0.3f32],
            class_names: vec![],
            apply_softmax: false,
            topk: 5,
        }
    }
}

impl_model_config_methods!(ImageClassificationConfig, model);
impl_process_config_methods!(ImageClassificationConfig, processor);
