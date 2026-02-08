use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{
    inputs, Config, DynConf, Engine, Engines, FromConfig, Hbb, Image, ImageProcessor, Model,
    Module, ResizeModeType, Xs, X, Y,
};

/// RT-DETR: Real-Time Detection Transformer
#[derive(Debug, Builder)]
pub struct RTDETR {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub names: Vec<String>,
    pub confs: DynConf,
    pub processor: ImageProcessor,
    pub spec: String,
    pub classes_excluded: Vec<usize>,
    pub classes_retained: Vec<usize>,
}

impl Model for RTDETR {
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
        let classes_excluded = config.inference.classes_excluded;
        let classes_retained = config.inference.classes_retained;
        if !classes_excluded.is_empty() {
            tracing::info!("classes_excluded: {classes_excluded:?}");
        }
        if !classes_retained.is_empty() {
            tracing::info!("classes_retained: {classes_retained:?}");
        }
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
            classes_excluded,
            classes_retained,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        let x1 = crate::perf!("RTDETR::preprocess", self.processor.process(images)?);
        let x2 = X::from(vec![self.height as f32, self.width as f32])
            .insert_axis(0)?
            .repeat(0, self.batch)?;
        let ys = crate::perf!(
            "RTDETR::inference",
            engines.run(&Module::Model, inputs![&x1, x2]?)?
        );
        crate::perf!("RTDETR::postprocess", self.postprocess(&ys))
    }
}

impl RTDETR {
    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let resize_mode = match self.processor.resize_mode_type() {
            Some(ResizeModeType::Letterbox) => ResizeModeType::Letterbox,
            Some(ResizeModeType::FitAdaptive) => ResizeModeType::FitAdaptive,
            Some(ResizeModeType::FitExact) => ResizeModeType::FitExact,
            Some(x) => anyhow::bail!("Unsupported resize mode for RTDETR postprocess: {x:?}. Supported: FitExact, FitAdaptive, Letterbox"),
            _ => anyhow::bail!("No resize mode specified. Supported: FitExact, FitAdaptive, Letterbox"),
        };

        let labels = outputs
            .get::<i64>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get labels"))?;
        let boxes = outputs
            .get::<f32>(1)
            .ok_or_else(|| anyhow::anyhow!("Failed to get bboxes"))?;
        let scores = outputs
            .get::<f32>(2)
            .ok_or_else(|| anyhow::anyhow!("Failed to get scores"))?;

        let ys: Vec<Y> = labels
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(boxes.axis_iter(Axis(0)).into_par_iter())
            .zip(scores.axis_iter(Axis(0)).into_par_iter())
            .enumerate()
            .filter_map(|(idx, ((labels, boxes), scores))| {
                let info = &self.processor.images_transform_info[idx];
                let (image_height, image_width) = (info.height_src, info.width_src);
                let y_bboxes: Vec<Hbb> = scores
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &score)| {
                        let class_id = labels[i] as usize;
                        if score < self.confs[class_id] {
                            return None;
                        }

                        if !self.classes_excluded.is_empty()
                            && self.classes_excluded.contains(&class_id)
                        {
                            return None;
                        }

                        if !self.classes_retained.is_empty()
                            && !self.classes_retained.contains(&class_id)
                        {
                            return None;
                        }

                        let xyxy_raw = boxes.slice(s![i, ..]);
                        let (x1, y1, x2, y2) = match resize_mode {
                            ResizeModeType::FitExact => {
                                let scale_x = image_width as f32 / self.width as f32;
                                let scale_y = image_height as f32 / self.height as f32;
                                (
                                    xyxy_raw[0] * scale_x,
                                    xyxy_raw[1] * scale_y,
                                    xyxy_raw[2] * scale_x,
                                    xyxy_raw[3] * scale_y,
                                )
                            }
                            ResizeModeType::Letterbox => {
                                let ratio = info.height_scale;
                                let pad_w = info.width_pad;
                                let pad_h = info.height_pad;
                                (
                                    (xyxy_raw[0] - pad_w) / ratio,
                                    (xyxy_raw[1] - pad_h) / ratio,
                                    (xyxy_raw[2] - pad_w) / ratio,
                                    (xyxy_raw[3] - pad_h) / ratio,
                                )
                            }
                            ResizeModeType::FitAdaptive => {
                                let ratio = info.height_scale;
                                (
                                    xyxy_raw[0] / ratio,
                                    xyxy_raw[1] / ratio,
                                    xyxy_raw[2] / ratio,
                                    xyxy_raw[3] / ratio,
                                )
                            }
                            _ => unreachable!(),
                        };

                        let x1 = x1.max(0.0).min(image_width as _);
                        let y1 = y1.max(0.0).min(image_height as _);
                        let x2 = x2.max(0.0).min(image_width as _);
                        let y2 = y2.max(0.0).min(image_height as _);
                        let mut hbb = Hbb::default()
                            .with_xyxy(x1, y1, x2, y2)
                            .with_confidence(score)
                            .with_id(class_id);
                        if !self.names.is_empty() {
                            hbb = hbb.with_name(&self.names[class_id]);
                        }
                        Some(hbb)
                    })
                    .collect();

                Some(Y::default().with_hbbs(&y_bboxes))
            })
            .collect();

        Ok(ys)
    }
}
