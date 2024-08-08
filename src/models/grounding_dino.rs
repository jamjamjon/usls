use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Axis};
use rayon::prelude::*;
use tokenizers::Tokenizer;

use crate::{auto_load, Bbox, DynConf, MinOptMax, Ops, Options, OrtEngine, Xs, X, Y};

#[derive(Debug)]
pub struct GroundingDINO {
    pub engine: OrtEngine,
    height: MinOptMax,
    width: MinOptMax,
    batch: MinOptMax,
    tokenizer: Tokenizer,
    pub context_length: usize,
    confs_visual: DynConf,
    confs_textual: DynConf,
}

impl GroundingDINO {
    pub fn new(options: Options) -> Result<Self> {
        let mut engine = OrtEngine::new(&options)?;
        let (batch, height, width) = (
            engine.inputs_minoptmax()[0][0].to_owned(),
            engine.inputs_minoptmax()[0][2].to_owned(),
            engine.inputs_minoptmax()[0][3].to_owned(),
        );
        let context_length = options.context_length.unwrap_or(256);
        // let special_tokens = ["[CLS]", "[SEP]", ".", "?"];
        let tokenizer = match options.tokenizer {
            Some(x) => x,
            None => match auto_load("tokenizer-groundingdino.json", Some("tokenizers")) {
                Err(err) => anyhow::bail!("No tokenizer's file found: {:?}", err),
                Ok(x) => x,
            },
        };
        let tokenizer = match Tokenizer::from_file(tokenizer) {
            Err(err) => anyhow::bail!("Failed to build tokenizer: {:?}", err),
            Ok(x) => x,
        };
        let confs_visual = DynConf::new(&options.confs, 1);
        let confs_textual = DynConf::new(&options.confs, 1);

        engine.dry_run()?;

        Ok(Self {
            engine,
            batch,
            height,
            width,
            tokenizer,
            context_length,
            confs_visual,
            confs_textual,
        })
    }

    pub fn run(&mut self, xs: &[DynamicImage], texts: &[&str]) -> Result<Vec<Y>> {
        // image embeddings
        let image_embeddings = X::apply(&[
            Ops::Letterbox(
                xs,
                self.height() as u32,
                self.width() as u32,
                "CatmullRom",
                114,
                "auto",
                false,
            ),
            Ops::Normalize(0., 255.),
            Ops::Standardize(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225], 3),
            Ops::Nhwc2nchw,
        ])?;

        // encoding
        let text = Self::parse_texts(texts);
        let encoding = match self.tokenizer.encode(text, true) {
            Err(err) => anyhow::bail!("{}", err),
            Ok(x) => x,
        };
        let tokens = encoding.get_tokens();

        // input_ids
        let input_ids = X::from(
            encoding
                .get_ids()
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<_>>(),
        )
        .insert_axis(0)?;

        // token_type_ids
        let token_type_ids = X::zeros(&[self.batch() as usize, input_ids.len()]);

        // attention_mask
        let attention_mask = X::ones(&[self.batch() as usize, input_ids.len()]);

        // position_ids
        let position_ids = encoding
            .get_tokens()
            .iter()
            .map(|x| if x == "." { 1. } else { 0. })
            .collect::<Vec<_>>();
        let mut position_ids_c = position_ids.clone();
        position_ids_c[0] = 1.;
        let n = position_ids_c.len();
        position_ids_c[n - 1] = 1.;
        let position_ids = X::from(position_ids).insert_axis(0)?;

        // text_self_attention_masks
        let mut text_self_attention_masks =
            Array::zeros((input_ids.len(), input_ids.len())).into_dyn();
        let mut i_last = -1;
        for (i, &v) in position_ids_c.iter().enumerate() {
            if v == 0. {
                if i_last == -1 {
                    i_last = i as isize;
                } else {
                    i_last = -1;
                }
            } else if v == 1. {
                if i_last == -1 {
                    text_self_attention_masks.slice_mut(s![i, i]).fill(1.);
                } else {
                    text_self_attention_masks
                        .slice_mut(s![i_last as _..i + 1, i_last as _..i + 1])
                        .fill(1.);
                }
                i_last = -1;
            } else {
                continue;
            }
        }
        let text_self_attention_masks = X::from(text_self_attention_masks).insert_axis(0)?;

        // run
        let ys = self.engine.run(Xs::from(vec![
            image_embeddings,
            input_ids,
            attention_mask,
            position_ids,
            token_type_ids,
            text_self_attention_masks,
        ]))?;

        // post-process
        self.postprocess(ys, xs, tokens)
    }

    fn postprocess(&self, xs: Xs, xs0: &[DynamicImage], tokens: &[String]) -> Result<Vec<Y>> {
        let ys: Vec<Y> = xs["logits"]
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, logits)| {
                let image_width = xs0[idx].width() as f32;
                let image_height = xs0[idx].height() as f32;
                let ratio =
                    (self.width() as f32 / image_width).min(self.height() as f32 / image_height);

                let y_bboxes: Vec<Bbox> = logits
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .filter_map(|(i, clss)| {
                        let (class_id, &conf) = clss
                            .mapv(|x| 1. / ((-x).exp() + 1.))
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))?;

                        if conf < self.conf_visual() {
                            return None;
                        }

                        let bbox = xs["boxes"].slice(s![idx, i, ..]).mapv(|x| x / ratio);
                        let cx = bbox[0] * self.width() as f32;
                        let cy = bbox[1] * self.height() as f32;
                        let w = bbox[2] * self.width() as f32;
                        let h = bbox[3] * self.height() as f32;
                        let x = cx - w / 2.;
                        let y = cy - h / 2.;
                        let x = x.max(0.0).min(image_width);
                        let y = y.max(0.0).min(image_height);

                        Some(
                            Bbox::default()
                                .with_xywh(x, y, w, h)
                                .with_id(class_id as _)
                                .with_name(&tokens[class_id])
                                .with_confidence(conf),
                        )
                    })
                    .collect();

                if !y_bboxes.is_empty() {
                    Some(Y::default().with_bboxes(&y_bboxes))
                } else {
                    None
                }
            })
            .collect();
        Ok(ys)
    }

    pub fn parse_texts(texts: &[&str]) -> String {
        let mut y = String::new();
        for text in texts.iter() {
            if !text.is_empty() {
                y.push_str(&format!("{} . ", text));
            }
        }
        y
    }

    pub fn conf_visual(&self) -> f32 {
        self.confs_visual[0]
    }

    pub fn conf_textual(&self) -> f32 {
        self.confs_textual[0]
    }

    pub fn batch(&self) -> isize {
        self.batch.opt
    }

    pub fn width(&self) -> isize {
        self.width.opt
    }

    pub fn height(&self) -> isize {
        self.height.opt
    }
}
