use aksr::Builder;
use anyhow::Result;
use ndarray::Axis;
use rayon::prelude::*;

use crate::{
    Config, DynConf, Engine, Engines, FromConfig, Hbb, Image, ImageProcessor, Keypoint, Model,
    Module, ResizeModeType, Xs, X, Y,
};

/// RTMO: Real-Time Multi-Object Detection
#[derive(Builder, Debug)]
pub struct RTMO {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub spec: String,
    pub processor: ImageProcessor,
    pub confs: DynConf,
    pub kconfs: DynConf,
    pub names: Vec<String>,
}

impl Model for RTMO {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let spec = engine.spec().to_string();
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&512.into()).opt(),
            engine.try_width().unwrap_or(&512.into()).opt(),
        );
        let names: Vec<String> = config.inference.keypoint_names;
        let nk = config.inference.num_keypoints.unwrap_or(17);
        let confs = DynConf::new_or_default(&config.inference.class_confs, 1);
        let kconfs = DynConf::new_or_default(&config.inference.keypoint_confs, nk);
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            height,
            width,
            batch,
            spec,
            processor,
            confs,
            kconfs,
            names,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let x = crate::perf!("RTMO::preprocess", self.processor.process(images)?);
        let ys = crate::perf!("RTMO::inference", engines.run(&Module::Model, &x)?);
        crate::perf!("RTMO::postprocess", self.postprocess(&ys))
    }
}

impl RTMO {
    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let resize_mode = match self.processor.resize_mode_type() {
            Some(ResizeModeType::Letterbox) => ResizeModeType::Letterbox,
            Some(ResizeModeType::FitAdaptive) => ResizeModeType::FitAdaptive,
            Some(ResizeModeType::FitExact) => ResizeModeType::FitExact,
            Some(x) => anyhow::bail!("Unsupported resize mode for RTMO postprocess: {x:?}. Supported: FitExact, FitAdaptive, Letterbox"),
            _ => anyhow::bail!("No resize mode specified. Supported: FitExact, FitAdaptive, Letterbox"),
        };

        let x0 = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output 0"))?;
        let x1 = outputs
            .get::<f32>(1)
            .ok_or_else(|| anyhow::anyhow!("Failed to get output 1"))?;
        let xs = (X::from(x0), X::from(x1));
        let ys: Vec<Y> =
            xs.0.axis_iter(Axis(0))
                .into_par_iter()
                .zip(xs.1.axis_iter(Axis(0)).into_par_iter())
                .enumerate()
                .map(|(idx, (batch_bboxes, batch_kpts))| {
                    let info = &self.processor.images_transform_info[idx];
                    let (height_original, width_original) = (info.height_src, info.width_src);

                    let mut y_bboxes = Vec::new();
                    let mut y_kpts: Vec<Vec<Keypoint>> = Vec::new();
                    for (xyxyc, kpts) in batch_bboxes
                        .axis_iter(Axis(0))
                        .zip(batch_kpts.axis_iter(Axis(0)))
                    {
                        // Transform bbox coordinates based on resize mode
                        let (x1, y1, x2, y2) = match resize_mode {
                            ResizeModeType::FitExact => {
                                let scale_x = width_original as f32 / self.width as f32;
                                let scale_y = height_original as f32 / self.height as f32;
                                (
                                    xyxyc[0] * scale_x,
                                    xyxyc[1] * scale_y,
                                    xyxyc[2] * scale_x,
                                    xyxyc[3] * scale_y,
                                )
                            }
                            ResizeModeType::Letterbox => {
                                let ratio = info.height_scale;
                                let pad_w = info.width_pad;
                                let pad_h = info.height_pad;
                                (
                                    (xyxyc[0] - pad_w) / ratio,
                                    (xyxyc[1] - pad_h) / ratio,
                                    (xyxyc[2] - pad_w) / ratio,
                                    (xyxyc[3] - pad_h) / ratio,
                                )
                            }
                            ResizeModeType::FitAdaptive => {
                                let ratio = info.height_scale;
                                (
                                    xyxyc[0] / ratio,
                                    xyxyc[1] / ratio,
                                    xyxyc[2] / ratio,
                                    xyxyc[3] / ratio,
                                )
                            }
                            _ => unreachable!(),
                        };
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

                        // keypoints - transform based on resize mode
                        let mut kpts_ = Vec::new();
                        for (i, kpt) in kpts.axis_iter(Axis(0)).enumerate() {
                            let (x, y) = match resize_mode {
                                ResizeModeType::FitExact => {
                                    let scale_x = width_original as f32 / self.width as f32;
                                    let scale_y = height_original as f32 / self.height as f32;
                                    (kpt[0] * scale_x, kpt[1] * scale_y)
                                }
                                ResizeModeType::Letterbox => {
                                    let ratio = info.height_scale;
                                    let pad_w = info.width_pad;
                                    let pad_h = info.height_pad;
                                    ((kpt[0] - pad_w) / ratio, (kpt[1] - pad_h) / ratio)
                                }
                                ResizeModeType::FitAdaptive => {
                                    let ratio = info.height_scale;
                                    (kpt[0] / ratio, kpt[1] / ratio)
                                }
                                _ => unreachable!(),
                            };
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
