use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Array2, Axis};
use rayon::prelude::*;
use std::fmt::Write;

use crate::{
    elapsed_module, inputs, Config, DynConf, Engine, Engines, FromConfig, Hbb, Image,
    ImageProcessor, Model, Module, TextProcessor, Xs, X, Y,
};

/// Grounding DINO model for open-vocabulary object detection.
#[derive(Builder, Debug)]
pub struct GroundingDINO {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub confs_visual: DynConf,
    pub confs_textual: DynConf,
    pub class_names: Vec<String>,
    pub class_ids_map: Vec<Option<usize>>,
    pub tokens: Vec<String>,
    pub token_ids: Vec<f32>,
    pub image_processor: ImageProcessor,
    pub text_processor: TextProcessor,
    pub spec: String,
    pub token_level_class: bool,
}

impl Model for GroundingDINO {
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
            engine.try_height().unwrap_or(&800.into()).opt(),
            engine.try_width().unwrap_or(&1200.into()).opt(),
        );
        let class_names: Vec<_> = config
            .inference
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
        let token_level_class = config.inference.token_level_class;
        let text_prompt = class_names.iter().fold(String::new(), |mut acc, text| {
            write!(&mut acc, "{text}.").unwrap();
            acc
        });
        let class_confs = config.class_confs().to_vec();
        let text_confs = config.text_confs().to_vec();
        let image_processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        #[cfg(feature = "vlm")]
        let text_processor = TextProcessor::from_config(config.text_processor)?;
        #[cfg(not(feature = "vlm"))]
        let text_processor = TextProcessor::default();
        let token_ids = text_processor.encode_text_ids(&text_prompt, true)?;
        let tokens = text_processor.encode_text_tokens(&text_prompt, true)?;

        // classes
        let (class_names, class_ids_map, confs_visual, confs_textual) = if token_level_class {
            // Token-level - each valid token is a separate class
            let (token_class_names, token_class_ids_map) =
                Self::process_token_level_classes(&tokens);
            let num_token_classes = token_class_names.len();
            let confs_visual = DynConf::new_or_default(&class_confs, num_token_classes);
            let confs_textual = DynConf::new_or_default(&text_confs, num_token_classes);
            (
                token_class_names,
                token_class_ids_map,
                confs_visual,
                confs_textual,
            )
        } else {
            // Phrase-level - original behavior
            let class_ids_map = Self::process_phrase_level_classes(&tokens);
            let confs_visual = DynConf::new_or_default(&class_confs, class_names.len());
            let confs_textual = DynConf::new_or_default(&text_confs, class_names.len());
            (class_names, class_ids_map, confs_visual, confs_textual)
        };

        let model = Self {
            batch,
            height,
            width,
            confs_visual,
            confs_textual,
            class_names,
            token_ids,
            tokens,
            image_processor,
            text_processor,
            spec,
            class_ids_map,
            token_level_class,
        };

        let engines = Engines::from(engine);
        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        // Preprocess
        let image_embeddings = elapsed_module!("GroundingDINO", "preprocess", {
            self.batch = images.len();
            self.image_processor.process(images)?
        });

        let input_ids = X::from(self.token_ids.clone())
            .insert_axis(0)?
            .repeat(0, self.batch)?;
        let (text_self_attention_masks, position_ids) =
            Self::gen_text_attn_masks_and_pos_ids(&self.token_ids)?;
        let text_self_attention_masks = text_self_attention_masks
            .insert_axis(0)?
            .repeat(0, self.batch)?;
        let position_ids = position_ids.insert_axis(0)?.repeat(0, self.batch)?;

        // Inference
        let ys = elapsed_module!(
            "GroundingDINO",
            "inference",
            engines.run(
                &Module::Model,
                inputs![
                    &image_embeddings,
                    input_ids.view(),
                    position_ids.view(),
                    text_self_attention_masks.view()
                ]?
            )?
        );

        // Postprocess
        elapsed_module!("GroundingDINO", "postprocess", self.postprocess(&ys))
    }
}

impl GroundingDINO {
    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let logits = outputs
            .get_by_name::<f32>("logits")
            .ok_or_else(|| anyhow::anyhow!("Failed to get logits"))?;
        let boxes = outputs
            .get_by_name::<f32>("boxes")
            .ok_or_else(|| anyhow::anyhow!("Failed to get boxes"))?;

        let ys: Vec<Y> = logits
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, logits_batch)| {
                let info = &self.image_processor.images_transform_info[idx];
                let (image_height, image_width, ratio) =
                    (info.height_src, info.width_src, info.height_scale);
                let y_bboxes: Vec<Hbb> = logits_batch
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .filter_map(|(i, clss)| {
                        let clss_probs = clss.mapv(|x| 1. / ((-x).exp() + 1.));
                        let (class_id, &max_conf) = clss_probs
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))?;

                        // Stage 1: box threshold - filter by max confidence
                        if max_conf < self.confs_visual[0] {
                            return None;
                        }

                        // Stage 2: text threshold - check token-level probability
                        // let token_conf = clss_probs[class_id]; // max_conf
                        let c = self.class_ids_map[class_id]?;
                        let token_conf = max_conf;
                        if token_conf < self.confs_textual[c] {
                            return None;
                        }

                        // Compute bbox coordinates
                        let bbox = boxes.slice(s![idx, i, ..]).mapv(|x| x / ratio);
                        let cx = bbox[0] * self.width as f32;
                        let cy = bbox[1] * self.height as f32;
                        let w = bbox[2] * self.width as f32;
                        let h = bbox[3] * self.height as f32;
                        let x = cx - w / 2.;
                        let y = cy - h / 2.;
                        let x = x.max(0.0).min(image_width as _);
                        let y = y.max(0.0).min(image_height as _);

                        // Use token-level confidence as the final score
                        // In token-level mode, display the actual token; in phrase mode, display the phrase
                        let name = if self.token_level_class {
                            &self.tokens[class_id]
                        } else {
                            &self.class_names[c]
                        };

                        Some(
                            Hbb::default()
                                .with_xywh(x, y, w, h)
                                .with_confidence(token_conf)
                                .with_name(name)
                                .with_id(c as _),
                        )
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

    /// Phrase-level classes (original behavior)
    /// Maps tokens to phrase indices, separated by '.'
    fn process_phrase_level_classes(tokens: &[String]) -> Vec<Option<usize>> {
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

    /// Token-level classes
    /// Each valid token becomes its own class
    fn process_token_level_classes(tokens: &[String]) -> (Vec<String>, Vec<Option<usize>>) {
        let mut class_names = Vec::new();
        let mut class_ids_map = Vec::with_capacity(tokens.len());

        for token in tokens {
            if token == "[CLS]" || token == "[SEP]" || token == "." {
                class_ids_map.push(None);
            } else {
                let class_id = class_names.len();
                class_names.push(token.clone());
                class_ids_map.push(Some(class_id));
            }
        }

        (class_names, class_ids_map)
    }
}
