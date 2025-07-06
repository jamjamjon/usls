use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Array2, Axis};
use rayon::prelude::*;
use std::fmt::Write;

use crate::{elapsed_module, Config, DynConf, Engine, Hbb, Image, Processor, Xs, X, Y};

#[derive(Builder, Debug)]
/// Grounding DINO model for open-vocabulary object detection.
pub struct GroundingDINO {
    pub engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    confs_visual: DynConf,
    confs_textual: DynConf,
    class_names: Vec<String>,
    class_ids_map: Vec<Option<usize>>,
    tokens: Vec<String>,
    token_ids: Vec<f32>,

    processor: Processor,
    spec: String,
}

impl GroundingDINO {
    pub fn new(config: Config) -> Result<Self> {
        let engine = Engine::try_from_config(&config.model)?;
        let spec = engine.spec().to_string();
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&800.into()).opt(),
            engine.try_width().unwrap_or(&1200.into()).opt(),
        );
        let class_names: Vec<_> = config
            .text_names
            .iter()
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        if class_names.is_empty() {
            anyhow::bail!(
                "No valid class names were provided in the config. Ensure the 'text_names' field is non-empty and contains valid class names."
            );
        }
        let text_prompt = class_names.iter().fold(String::new(), |mut acc, text| {
            write!(&mut acc, "{}.", text).unwrap();
            acc
        });
        let confs_visual = DynConf::new_or_default(config.class_confs(), class_names.len());
        let confs_textual = DynConf::new_or_default(config.text_confs(), class_names.len());
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let token_ids = processor.encode_text_ids(&text_prompt, true)?;
        let tokens = processor.encode_text_tokens(&text_prompt, true)?;
        let class_ids_map = Self::process_class_ids(&tokens);

        Ok(Self {
            engine,
            batch,
            height,
            width,
            confs_visual,
            confs_textual,
            class_names,
            token_ids,
            tokens,
            processor,
            spec,
            class_ids_map,
        })
    }

    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs> {
        // encode images
        let image_embeddings = self.processor.process_images(xs)?;

        // encode texts
        let input_ids = X::from(self.token_ids.clone())
            .insert_axis(0)?
            .repeat(0, self.batch)?;
        let token_type_ids = X::zeros(&[self.batch, self.tokens.len()]);
        let attention_mask = X::ones(&[self.batch, self.tokens.len()]);
        let (text_self_attention_masks, position_ids) =
            Self::gen_text_attn_masks_and_pos_ids(&self.token_ids)?;
        let text_self_attention_masks = text_self_attention_masks
            .insert_axis(0)?
            .repeat(0, self.batch)?;
        let position_ids = position_ids.insert_axis(0)?.repeat(0, self.batch)?;

        // inputs
        let xs = Xs::from(vec![
            image_embeddings,
            input_ids,
            attention_mask,
            position_ids,
            token_type_ids,
            text_self_attention_masks,
        ]);

        Ok(xs)
    }

    pub fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed_module!("GroundingDINO", "preprocess", self.preprocess(xs)?);
        let ys = elapsed_module!("GroundingDINO", "inference", self.inference(ys)?);
        let ys = elapsed_module!("GroundingDINO", "postprocess", self.postprocess(ys)?);

        Ok(ys)
    }
    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    fn postprocess(&self, xs: Xs) -> Result<Vec<Y>> {
        let ys: Vec<Y> = xs["logits"]
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, logits)| {
                let (image_height, image_width) = (
                    self.processor.images_transform_info[idx].height_src,
                    self.processor.images_transform_info[idx].width_src,
                );
                let ratio = self.processor.images_transform_info[idx].height_scale;
                let y_bboxes: Vec<Hbb> = logits
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .filter_map(|(i, clss)| {
                        let (class_id, &conf) = clss
                            .mapv(|x| 1. / ((-x).exp() + 1.))
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))?;

                        if conf < self.confs_visual[0] {
                            return None;
                        }

                        let bbox = xs["boxes"].slice(s![idx, i, ..]).mapv(|x| x / ratio);
                        let cx = bbox[0] * self.width as f32;
                        let cy = bbox[1] * self.height as f32;
                        let w = bbox[2] * self.width as f32;
                        let h = bbox[3] * self.height as f32;
                        let x = cx - w / 2.;
                        let y = cy - h / 2.;
                        let x = x.max(0.0).min(image_width as _);
                        let y = y.max(0.0).min(image_height as _);

                        self.class_ids_map[class_id].map(|c| {
                            let mut bbox =
                                Hbb::default().with_xywh(x, y, w, h).with_confidence(conf);

                            if conf > self.confs_textual[c] {
                                bbox = bbox.with_name(&self.class_names[c]).with_id(c as _);
                            }
                            bbox
                        })
                    })
                    .collect();

                if !y_bboxes.is_empty() {
                    Some(Y::default().with_hbbs(&y_bboxes))
                } else {
                    None
                }
            })
            .collect();

        Ok(ys)
    }

    fn gen_text_attn_masks_and_pos_ids(input_ids: &[f32]) -> Result<(X, X)> {
        let n = input_ids.len();
        let mut vs: Vec<f32> = input_ids
            .iter()
            .map(|&x| {
                if (x - 101.0).abs() < f32::EPSILON
                    || (x - 1012.0).abs() < f32::EPSILON
                    || (x - 102.0).abs() < f32::EPSILON
                {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        vs[0] = 1.0;
        vs[n - 1] = 1.0;

        let special_idxs: Vec<usize> = vs
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                if (v - 1.0).abs() < f32::EPSILON {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let mut attn = Array2::<f32>::eye(n);
        let mut pos_ids = vec![0f32; n];
        let mut prev = special_idxs[0];
        for &idx in special_idxs.iter() {
            if idx == 0 || idx == n - 1 {
            } else {
                for r in (prev + 1)..=idx {
                    for c in (prev + 1)..=idx {
                        attn[[r, c]] = 1.0;
                    }
                }
                for (offset, pos_id) in pos_ids[prev + 1..=idx].iter_mut().enumerate() {
                    *pos_id = offset as f32;
                }
            }
            prev = idx;
        }

        Ok((X::from(attn.into_dyn()), X::from(pos_ids)))
    }

    fn process_class_ids(tokens: &[String]) -> Vec<Option<usize>> {
        let mut result = Vec::with_capacity(tokens.len());
        let mut idx = 0;
        for token in tokens {
            if token == "[CLS]" || token == "[SEP]" {
                result.push(None);
            } else if token == "." {
                result.push(None);
                idx += 1;
            } else {
                result.push(Some(idx));
            }
        }
        result
    }
}
