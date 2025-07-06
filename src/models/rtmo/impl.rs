use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{elapsed_module, Config, DynConf, Engine, Hbb, Image, Keypoint, Processor, Xs, Y};

#[derive(Builder, Debug)]
pub struct RTMO {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,

    spec: String,
    processor: Processor,
    confs: DynConf,
    kconfs: DynConf,
    names: Vec<String>,
}

impl RTMO {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let spec = engine.spec().to_string();
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&512.into()).opt(),
            engine.try_width().unwrap_or(&512.into()).opt(),
        );
        let names: Vec<String> = config.keypoint_names().to_vec();
        let nk = config.nk().unwrap_or(17);
        let confs = DynConf::new_or_default(config.class_confs(), 1);
        let kconfs = DynConf::new_or_default(config.keypoint_confs(), nk);
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            engine,
            height,
            width,
            batch,

            spec,
            processor,
            confs,
            kconfs,
            names,
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        Ok(self.processor.process_images(xs)?.into())
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("RTMO", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("RTMO", "inference", self.inference(ys)?);
        let ys = elapsed_module!("RTMO", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }

    fn postprocess(&mut self, xs: Xs) -> Result<Vec<Y>> {
        let ys: Vec<Y> = xs[0]
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(xs[1].axis_iter(Axis(0)).into_par_iter())
            .enumerate()
            .map(|(idx, (batch_bboxes, batch_kpts))| {
                let (height_original, width_original) = (
                    self.processor.images_transform_info[idx].height_src,
                    self.processor.images_transform_info[idx].width_src,
                );
                let ratio = self.processor.images_transform_info[idx].height_scale;

                let mut y_bboxes = Vec::new();
                let mut y_kpts: Vec<Vec<Keypoint>> = Vec::new();
                for (xyxyc, kpts) in batch_bboxes
                    .axis_iter(Axis(0))
                    .zip(batch_kpts.axis_iter(Axis(0)))
                {
                    // bbox
                    let x1 = xyxyc[0] / ratio;
                    let y1 = xyxyc[1] / ratio;
                    let x2 = xyxyc[2] / ratio;
                    let y2 = xyxyc[3] / ratio;
                    let confidence = xyxyc[4];

                    if confidence < self.confs[0] {
                        continue;
                    }
                    y_bboxes.push(
                        Hbb::default()
                            .with_xyxy(
                                x1.max(0.0f32).min(width_original as _),
                                y1.max(0.0f32).min(height_original as _),
                                x2,
                                y2,
                            )
                            .with_confidence(confidence)
                            .with_id(0)
                            .with_name("Person"),
                    );

                    // keypoints
                    let mut kpts_ = Vec::new();
                    for (i, kpt) in kpts.axis_iter(Axis(0)).enumerate() {
                        let x = kpt[0] / ratio;
                        let y = kpt[1] / ratio;
                        let c = kpt[2];
                        if c < self.kconfs[i] {
                            kpts_.push(Keypoint::default());
                        } else {
                            let mut kpt_ =
                                Keypoint::default().with_id(i).with_confidence(c).with_xy(
                                    x.max(0.0f32).min(width_original as _),
                                    y.max(0.0f32).min(height_original as _),
                                );

                            if !self.names.is_empty() {
                                kpt_ = kpt_.with_name(&self.names[i]);
                            }

                            kpts_.push(kpt_);
                        }
                    }
                    y_kpts.push(kpts_);
                }
                Y::default().with_hbbs(&y_bboxes).with_keypointss(&y_kpts)
            })
            .collect();

        Ok(ys)
    }
}
