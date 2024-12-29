use aksr::Builder;
use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Axis};
use rayon::prelude::*;

use crate::{elapsed, Bbox, DynConf, Engine, Options, Processor, Ts, Xs, Ys, X, Y};

#[derive(Builder, Debug)]
pub struct GroundingDINO {
    pub engine: Engine,
    height: usize,
    width: usize,
    batch: usize,
    confs_visual: DynConf,
    confs_textual: DynConf,
    class_names: Vec<String>,
    tokens: Vec<String>,
    token_ids: Vec<f32>,
    ts: Ts,
    processor: Processor,
    spec: String,
}

impl GroundingDINO {
    pub fn new(options: Options) -> Result<Self> {
        let engine = options.to_engine()?;
        let spec = engine.spec().to_string();

        let (batch, height, width, ts) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&800.into()).opt(),
            engine.try_width().unwrap_or(&1200.into()).opt(),
            engine.ts().clone(),
        );
        let processor = options
            .to_processor()?
            .with_image_width(width as _)
            .with_image_height(height as _);
        let confs_visual = DynConf::new(options.class_confs(), 1);
        let confs_textual = DynConf::new(options.text_confs(), 1);

        let class_names = Self::parse_texts(
            &options
                .text_names
                .expect("No class names specified!")
                .iter()
                .map(|x| x.as_str())
                .collect::<Vec<_>>(),
        );
        let token_ids = processor.encode_text_ids(&class_names, true)?;
        let tokens = processor.encode_text_tokens(&class_names, true)?;
        let class_names = tokens.clone();

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
            ts,
            processor,
            spec,
        })
    }

    fn preprocess(&mut self, xs: &[DynamicImage]) -> Result<Xs> {
        // encode images
        let image_embeddings = self.processor.process_images(xs)?;

        // encode texts
        let tokens_f32 = self
            .tokens
            .iter()
            .map(|x| if x == "." { 1. } else { 0. })
            .collect::<Vec<_>>();

        // input_ids
        let input_ids = X::from(self.token_ids.clone())
            .insert_axis(0)?
            .repeat(0, self.batch)?;

        // token_type_ids
        let token_type_ids = X::zeros(&[self.batch, tokens_f32.len()]);

        // attention_mask
        let attention_mask = X::ones(&[self.batch, tokens_f32.len()]);

        // text_self_attention_masks
        let text_self_attention_masks = Self::gen_text_self_attention_masks(&tokens_f32)?
            .insert_axis(0)?
            .repeat(0, self.batch)?;

        // position_ids
        let position_ids = X::from(tokens_f32).insert_axis(0)?.repeat(0, self.batch)?;

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

    pub fn forward(&mut self, xs: &[DynamicImage]) -> Result<Ys> {
        let ys = elapsed!("preprocess", self.ts, { self.preprocess(xs)? });
        let ys = elapsed!("inference", self.ts, { self.inference(ys)? });
        let ys = elapsed!("postprocess", self.ts, { self.postprocess(ys)? });

        Ok(ys)
    }

    pub fn summary(&mut self) {
        self.ts.summary();
    }

    fn inference(&mut self, xs: Xs) -> Result<Xs> {
        self.engine.run(xs)
    }

    fn postprocess(&self, xs: Xs) -> Result<Ys> {
        let ys: Vec<Y> = xs["logits"]
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter_map(|(idx, logits)| {
                let (image_height, image_width) = self.processor.image0s_size[idx];
                let ratio = self.processor.scale_factors_hw[idx][0];

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

                        Some(
                            Bbox::default()
                                .with_xywh(x, y, w, h)
                                .with_id(class_id as _)
                                .with_name(&self.class_names[class_id])
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

        Ok(ys.into())
    }

    fn parse_texts(texts: &[&str]) -> String {
        let mut y = String::new();
        for text in texts.iter() {
            if !text.is_empty() {
                y.push_str(&format!("{} . ", text));
            }
        }
        y
    }

    fn gen_text_self_attention_masks(tokens: &[f32]) -> Result<X> {
        let mut vs = tokens.to_vec();
        let n = vs.len();
        vs[0] = 1.;
        vs[n - 1] = 1.;
        let mut ys = Array::zeros((n, n)).into_dyn();
        let mut i_last = -1;
        for (i, &v) in vs.iter().enumerate() {
            if v == 0. {
                if i_last == -1 {
                    i_last = i as isize;
                } else {
                    i_last = -1;
                }
            } else if v == 1. {
                if i_last == -1 {
                    ys.slice_mut(s![i, i]).fill(1.);
                } else {
                    ys.slice_mut(s![i_last as _..i + 1, i_last as _..i + 1])
                        .fill(1.);
                }
                i_last = -1;
            } else {
                continue;
            }
        }
        Ok(X::from(ys))
    }
}
