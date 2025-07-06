use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;

use crate::{elapsed_module, Config, DynConf, Engine, Hbb, Image, Processor, Xs, X, Y};

/// OWL-ViT v2 model for open-vocabulary object detection.
#[derive(Debug, Builder)]
pub struct OWLv2 {
    engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    names: Vec<String>,
    names_with_prompt: Vec<String>,
    confs: DynConf,
    processor: Processor,
    spec: String,
    input_ids: X,
    attention_mask: X,
}

impl OWLv2 {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&960.into()).opt(),
            engine.try_width().unwrap_or(&960.into()).opt(),
        );
        let spec = engine.spec().to_owned();
        let names: Vec<String> = config.text_names().to_vec();
        if names.is_empty() {
            anyhow::bail!(
                "No valid class names were provided in the config. Ensure the 'text_names' field is non-empty and contains valid class names."
            );
        }
        let names_with_prompt: Vec<String> =
            names.iter().map(|x| format!("a photo of {}", x)).collect();
        let n = names.len();
        let confs = DynConf::new_or_default(config.class_confs(), n);
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let input_ids: Vec<f32> = processor
            .encode_texts_ids(
                &names_with_prompt
                    .iter()
                    .map(|x| x.as_str())
                    .collect::<Vec<_>>(),
                false,
            )?
            .into_iter()
            .flatten()
            .collect();
        let input_ids: X = ndarray::Array2::from_shape_vec((n, input_ids.len() / n), input_ids)?
            .into_dyn()
            .into();
        let attention_mask = X::ones_like(&input_ids);

        Ok(Self {
            engine,
            height,
            width,
            batch,
            spec,
            names,
            names_with_prompt,
            confs,
            processor,
            input_ids,
            attention_mask,
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        let image_embeddings = self.processor.process_images(xs)?;
        let xs = Xs::from(vec![
            self.input_ids.clone(),
            image_embeddings,
            self.attention_mask.clone(),
        ]);

        Ok(xs)
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("OWLv2", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("OWLv2", "inference", self.inference(ys)?);
        let ys = elapsed_module!("OWLv2", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }

    fn postprocess(&mut self, xs: Xs) -> Result<Vec<Y>> {
        let ys: Vec<Y> = xs[0]
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(xs[1].axis_iter(Axis(0)).into_par_iter())
            .enumerate()
            .filter_map(|(idx, (clss, bboxes))| {
                let (image_height, image_width) = (
                    self.processor.images_transform_info[idx].height_src,
                    self.processor.images_transform_info[idx].width_src,
                );

                let ratio = image_height.max(image_width) as f32;
                let y_bboxes: Vec<Hbb> = clss
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .filter_map(|(i, clss_)| {
                        let (class_id, &confidence) = clss_
                            .into_iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))?;

                        let confidence = 1. / ((-confidence).exp() + 1.);
                        if confidence < self.confs[class_id] {
                            return None;
                        }

                        let bbox = bboxes.slice(s![i, ..]).mapv(|x| x * ratio);
                        let (x, y, w, h) = (
                            (bbox[0] - bbox[2] / 2.).max(0.0f32),
                            (bbox[1] - bbox[3] / 2.).max(0.0f32),
                            bbox[2],
                            bbox[3],
                        );

                        Some(
                            Hbb::default()
                                .with_xywh(x, y, w, h)
                                .with_confidence(confidence)
                                .with_id(class_id)
                                .with_name(&self.names[class_id]),
                        )
                    })
                    .collect();

                Some(Y::default().with_hbbs(&y_bboxes))
            })
            .collect();

        Ok(ys)
    }
}
