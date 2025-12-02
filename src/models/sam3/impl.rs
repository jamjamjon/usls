use aksr::Builder;
use anyhow::Result;
use lru::LruCache;
use ndarray::{s, Array1};
use rayon::prelude::*;
use std::num::NonZeroUsize;

use crate::{
    elapsed_module, Config, DynConf, Engine, Hbb, Image, Mask, Ops, Processor, Sam3Prompt, Xs, X, Y,
};

/// SAM3: Segment Anything with Concepts
#[derive(Builder, Debug)]
pub struct SAM3 {
    visual_encoder: Engine,
    textual_encoder: Engine,
    geometry_encoder: Engine,
    decoder: Engine,
    vision_batch: usize,
    text_batch: usize,
    geo_batch: usize,
    decoder_batch: usize,
    processor: Processor,
    conf: DynConf,
    text_cache: LruCache<String, (X, X)>,
    names: Vec<String>,
    spec: String,
}

impl SAM3 {
    pub fn new(config: Config) -> Result<Self> {
        let visual_encoder = Engine::try_from_config(&config.visual_encoder)?;
        let textual_encoder = Engine::try_from_config(&config.textual_encoder)?;
        let geometry_encoder = Engine::try_from_config(&config.encoder)?;
        let decoder = Engine::try_from_config(&config.decoder)?;
        let vision_batch = visual_encoder.batch().opt();
        let text_batch = textual_encoder.batch().opt();
        let geo_batch = geometry_encoder.batch().opt();
        let decoder_batch = decoder.batch().opt();
        let height = visual_encoder.try_height().unwrap_or(&1008.into()).opt();
        let width = visual_encoder.try_width().unwrap_or(&1008.into()).opt();
        let conf = DynConf::new_or_default(config.class_confs(), 1);
        let processor = Processor::try_from_config(&config.processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        Ok(Self {
            visual_encoder,
            textual_encoder,
            geometry_encoder,
            decoder,
            processor,
            conf,
            vision_batch,
            text_batch,
            geo_batch,
            decoder_batch,
            text_cache: LruCache::new(NonZeroUsize::new(100).unwrap()),
            names: vec![],
            spec: "sam3".to_string(),
        })
    }

    fn encode_images(&mut self, xs: &[Image]) -> Result<Xs> {
        let xs_ = self.processor.process_images(xs)?;
        self.visual_encoder.run(Xs::from(xs_))
    }

    fn encode_texts(&mut self, texts: &[&str]) -> Result<Vec<(X, X)>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let mut all_results = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.text_batch) {
            let n = chunk.len();
            let encodings = self.processor.encode_texts(chunk, true)?;
            let seq_len = encodings[0].get_ids().len();
            let input_ids = X::from_shape_vec(
                &[n, seq_len],
                encodings
                    .iter()
                    .flat_map(|e| e.get_ids().iter().map(|&id| id as f32))
                    .collect(),
            )?;
            let attention_mask = X::from_shape_vec(
                &[n, seq_len],
                encodings
                    .iter()
                    .flat_map(|e| e.get_attention_mask().iter().map(|&m| m as f32))
                    .collect(),
            )?;

            let ys = self
                .textual_encoder
                .run(Xs::from(vec![input_ids, attention_mask]))?;

            for i in 0..n {
                all_results.push((
                    ys[0]
                        .slice(s![i..i + 1, .., ..])
                        .to_owned()
                        .into_dyn()
                        .into(),
                    ys[1].slice(s![i..i + 1, ..]).to_owned().into_dyn().into(),
                ));
            }
        }

        Ok(all_results)
    }

    fn encode_geometry_batch(
        &mut self,
        prompts_boxes: &[Vec<[f32; 4]>],
        prompts_labels: &[Vec<i64>],
        fpn_feat: &X,
        fpn_pos: &X,
    ) -> Result<Vec<(X, X)>> {
        let total = prompts_boxes.len();
        if total == 0 {
            return Ok(vec![]);
        }

        // Check if all prompts have no boxes
        let max_boxes_all = prompts_boxes.iter().map(|b| b.len()).max().unwrap_or(0);
        if max_boxes_all == 0 {
            return Ok(vec![(X::zeros(&[1, 1, 256]), X::zeros(&[1, 1])); total]);
        }

        // Process in chunks based on geo_batch
        let mut all_results = Vec::with_capacity(total);

        for (chunk_start, chunk) in prompts_boxes
            .chunks(self.geo_batch)
            .enumerate()
            .map(|(i, c)| (i * self.geo_batch, c))
        {
            let chunk_size = chunk.len();
            let chunk_labels = &prompts_labels[chunk_start..chunk_start + chunk_size];

            // Find max boxes in this chunk
            let max_boxes = chunk.iter().map(|b| b.len()).max().unwrap_or(0);
            if max_boxes == 0 {
                all_results.extend(vec![
                    (X::zeros(&[1, 1, 256]), X::zeros(&[1, 1]));
                    chunk_size
                ]);
                continue;
            }

            // Pad all prompts to same number of boxes
            let mut boxes_flat = Vec::with_capacity(chunk_size * max_boxes * 4);
            let mut labels_flat = Vec::with_capacity(chunk_size * max_boxes);

            for (boxes, labels) in chunk.iter().zip(chunk_labels.iter()) {
                for box_ in boxes {
                    boxes_flat.extend_from_slice(box_);
                }
                for _ in boxes.len()..max_boxes {
                    boxes_flat.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
                }

                labels_flat.extend(labels.iter().map(|&l| l as f32));
                labels_flat.resize(labels_flat.len() + max_boxes - labels.len(), 0.0);
            }

            let input_boxes = X::from_shape_vec(&[chunk_size, max_boxes, 4], boxes_flat)?;
            let input_labels = X::from_shape_vec(&[chunk_size, max_boxes], labels_flat)?;

            // Expand fpn features to chunk batch dimension
            let fpn_feat_batch = fpn_feat.clone().broadcast([chunk_size, 256, 72, 72])?;
            let fpn_pos_batch = fpn_pos.clone().broadcast([chunk_size, 256, 72, 72])?;

            let ys = self.geometry_encoder.run(Xs::from(vec![
                input_boxes,
                input_labels,
                fpn_feat_batch,
                fpn_pos_batch,
            ]))?;

            // Split chunk results
            let geo_features = &ys[0];
            let geo_masks = &ys[1];

            for (i, boxes) in chunk.iter().enumerate() {
                let num_boxes = boxes.len();
                let feat = geo_features
                    .slice(ndarray::s![i..i + 1, ..num_boxes + 1, ..])
                    .to_owned()
                    .into_dyn()
                    .into();
                let mask = geo_masks
                    .slice(ndarray::s![i..i + 1, ..num_boxes + 1])
                    .to_owned()
                    .into_dyn()
                    .into();
                all_results.push((feat, mask));
            }
        }

        Ok(all_results)
    }

    fn decode(
        &mut self,
        fpn_features: &Xs,
        prompt_features: &[X],
        prompt_masks: &[X],
    ) -> Result<Vec<Xs>> {
        let n = prompt_features.len();
        if n == 0 {
            return Ok(vec![]);
        }

        // Concatenate prompts and broadcast FPN features
        let prompt_feat = X::concat(prompt_features, 0)?;
        let prompt_mask = X::concat(prompt_masks, 0)?;

        let (s0, s1, s2, s6) = (
            fpn_features[0].shape(),
            fpn_features[1].shape(),
            fpn_features[2].shape(),
            fpn_features[6].shape(),
        );

        let ys = self.decoder.run(Xs::from(vec![
            fpn_features[0]
                .clone()
                .broadcast([n, s0[1], s0[2], s0[3]])?,
            fpn_features[1]
                .clone()
                .broadcast([n, s1[1], s1[2], s1[3]])?,
            fpn_features[2]
                .clone()
                .broadcast([n, s2[1], s2[2], s2[3]])?,
            fpn_features[6]
                .clone()
                .broadcast([n, s6[1], s6[2], s6[3]])?,
            prompt_feat,
            prompt_mask,
        ]))?;

        // Split batch results
        Ok((0..n)
            .map(|i| {
                vec![
                    ys[0]
                        .slice(s![i..i + 1, .., .., ..])
                        .to_owned()
                        .into_dyn()
                        .into(),
                    ys[1]
                        .slice(s![i..i + 1, .., ..])
                        .to_owned()
                        .into_dyn()
                        .into(),
                    ys[2].slice(s![i..i + 1, ..]).to_owned().into_dyn().into(),
                    ys[3].slice(s![i..i + 1, ..]).to_owned().into_dyn().into(),
                ]
                .into()
            })
            .collect())
    }

    pub fn forward(&mut self, xs: &[Image], prompts: &[Sam3Prompt]) -> Result<Vec<Y>> {
        if xs.is_empty() || prompts.is_empty() {
            return Ok(vec![]);
        }

        // Update class names from prompts
        self.names = prompts.iter().map(|p| p.text.clone()).collect();

        // Step 1: Vision Encoding
        let all_fpn_features = elapsed_module!("SAM3", "vision-encoder", self.encode_images(xs)?);

        // Store original image dimensions
        let image_dims: Vec<(usize, usize)> = self
            .processor
            .images_transform_info
            .iter()
            .map(|info| (info.height_src as usize, info.width_src as usize))
            .collect();

        // Step 2: Text Encoding (with LRU cache)
        let uncached: Vec<&str> = prompts
            .iter()
            .map(|p| p.text.as_str())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .filter(|t| !self.text_cache.contains(&t.to_string()))
            .collect();

        if !uncached.is_empty() {
            for (text, feat) in uncached.iter().zip(elapsed_module!(
                "SAM3",
                "text-encoder",
                self.encode_texts(&uncached)?
            )) {
                self.text_cache.put(text.to_string(), feat);
            }
        }

        let text_features_batch: Vec<(X, X)> = prompts
            .iter()
            .map(|p| self.text_cache.get(&p.text).unwrap().clone())
            .collect();

        // Step 3: Process each image
        let mut results = Vec::with_capacity(xs.len());

        for (img_idx, (image_height, image_width)) in image_dims.iter().enumerate() {
            // Extract single image's FPN features
            let fpn_features: Xs = (0..8)
                .map(|i| {
                    let feat = &all_fpn_features[i];
                    feat.slice(ndarray::s![img_idx..img_idx + 1, .., .., ..])
                        .to_owned()
                        .into_dyn()
                        .into()
                })
                .collect::<Vec<X>>()
                .into();

            // Step 3.1: Prepare geometry data (single pass with unzip)
            let (prompts_boxes, prompts_labels, has_geometry_flags): (Vec<_>, Vec<_>, Vec<_>) =
                prompts
                    .iter()
                    .map(|p| {
                        if p.should_use_geometry() {
                            (
                                p.normalized_boxes(*image_width as f32, *image_height as f32),
                                p.labels.clone(),
                                true,
                            )
                        } else {
                            (vec![], vec![], false)
                        }
                    })
                    .fold(
                        (vec![], vec![], vec![]),
                        |(mut bs, mut ls, mut fs), (b, l, f)| {
                            bs.push(b);
                            ls.push(l);
                            fs.push(f);
                            (bs, ls, fs)
                        },
                    );

            // Step 3.2: Geometry Encoding
            let geo_features_batch = elapsed_module!("SAM3", "geometry-encoder", {
                self.encode_geometry_batch(
                    &prompts_boxes,
                    &prompts_labels,
                    &fpn_features[2],
                    &fpn_features[6],
                )?
            });

            // Step 3.3: Build prompt features (text + geometry concatenation)
            let prompt_features_all: Vec<_> = has_geometry_flags
                .iter()
                .zip(text_features_batch.iter())
                .enumerate()
                .map(|(idx, (has_geo, (text_features, text_mask)))| {
                    if *has_geo {
                        let (geo_features, geo_mask) = geo_features_batch[idx].clone();
                        (
                            X::concat(&[text_features.clone(), geo_features], 1).unwrap(),
                            X::concat(&[text_mask.clone(), geo_mask], 1).unwrap(),
                        )
                    } else {
                        (text_features.clone(), text_mask.clone())
                    }
                })
                .collect();

            // Step 3.4: Decoding (chunked by decoder_batch)
            let all_outputs: Vec<Xs> = elapsed_module!("SAM3", "decoder", {
                let mut outputs = Vec::with_capacity(prompts.len());
                for chunk in prompt_features_all.chunks(self.decoder_batch) {
                    let (features, masks): (Vec<_>, Vec<_>) = chunk.iter().cloned().unzip();
                    outputs.extend(self.decode(&fpn_features, &features, &masks)?);
                }
                outputs
            });

            // Step 3.5: Parallel postprocess with rayon
            let post_results: Vec<_> = all_outputs
                .iter()
                .zip(prompts.iter())
                .enumerate()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(prompt_idx, (outputs, prompt))| {
                    self.postprocess(
                        outputs,
                        *image_height,
                        *image_width,
                        prompt_idx,
                        prompt.class_name(),
                    )
                })
                .collect();

            // Combine results
            let mut combined_masks: Vec<Mask> = Vec::new();
            let mut combined_hbbs: Vec<Hbb> = Vec::new();

            for y in post_results.into_iter().flatten() {
                combined_masks.extend(y.masks().iter().cloned());
                combined_hbbs.extend(y.hbbs().iter().cloned());
            }

            results.push(
                Y::default()
                    .with_masks(&combined_masks)
                    .with_hbbs(&combined_hbbs),
            );
        }

        Ok(results)
    }

    fn postprocess(
        &self,
        outputs: &Xs,
        image_height: usize,
        image_width: usize,
        class_id: usize,
        class_name: &str,
    ) -> Result<Y> {
        let pred_masks = &outputs[0];
        let pred_boxes = &outputs[1];
        let pred_logits = &outputs[2];
        let presence_logits = &outputs[3];

        let presence_score = 1.0 / (1.0 + (-presence_logits[[0, 0]]).exp());
        let scores: Array1<f32> = pred_logits
            .slice(s![0, ..])
            .mapv(|x| 1.0 / (1.0 + (-x).exp()) * presence_score);

        let threshold = self.conf[0];

        let valid_indices: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|(_, &score)| score >= threshold)
            .map(|(idx, _)| idx)
            .collect();

        if valid_indices.is_empty() {
            return Ok(Y::default());
        }

        let mask_data: Vec<_> = valid_indices
            .iter()
            .map(|&idx| {
                let mask = pred_masks.slice(s![0, idx, .., ..]);
                let (mask_h, mask_w) = mask.dim();
                let mask_vec = mask.to_owned().into_raw_vec_and_offset().0;
                (idx, scores[idx], mask_vec, mask_h, mask_w)
            })
            .collect();

        let masks_and_boxes: Vec<_> = mask_data
            .into_par_iter()
            .filter_map(|(idx, score, mask_vec, mask_h, mask_w)| {
                let luma = Ops::resize_lumaf32_u8(
                    &mask_vec,
                    mask_w as _,
                    mask_h as _,
                    image_width as _,
                    image_height as _,
                    false,
                    "Bilinear",
                )
                .ok()?;

                let mask = Mask::new(&luma, image_width as u32, image_height as u32)
                    .ok()?
                    .with_id(class_id)
                    .with_name(class_name)
                    .with_confidence(score);

                let x1 = pred_boxes[[0, idx, 0]] * image_width as f32;
                let y1 = pred_boxes[[0, idx, 1]] * image_height as f32;
                let x2 = pred_boxes[[0, idx, 2]] * image_width as f32;
                let y2 = pred_boxes[[0, idx, 3]] * image_height as f32;
                let hbb = Hbb::default()
                    .with_xyxy(x1, y1, x2, y2)
                    .with_confidence(score)
                    .with_id(class_id)
                    .with_name(class_name);

                Some((mask, hbb))
            })
            .collect();

        let (y_masks, y_hbbs): (Vec<_>, Vec<_>) = masks_and_boxes.into_iter().unzip();

        Ok(Y::default().with_masks(&y_masks).with_hbbs(&y_hbbs))
    }

    pub fn with_cache_size(mut self, max_size: usize) -> Self {
        self.text_cache
            .resize(NonZeroUsize::new(max_size.max(1)).unwrap());
        self
    }

    pub fn clear_text_cache(&mut self) {
        self.text_cache.clear();
    }

    pub fn cache_stats(&self) {
        let n = self.text_cache.len();
        let cap = self.text_cache.cap().get();
        let dtype_size = self
            .textual_encoder
            .odtypes()
            .and_then(|d| d.first().map(|t| t.size_in_bytes()))
            .unwrap_or(4);
        // Text encoder outputs: (1, 32, 256) + (1, 32)
        let bytes_per_entry = (32 * 256 + 32) * dtype_size;
        let mem = crate::human_bytes_binary((n * bytes_per_entry) as f64, 2);
        let texts: Vec<_> = self.text_cache.iter().map(|(k, _)| k.as_str()).collect();
        println!("Text Cache: {n}/{cap}, ~{mem} | {texts:?}");
    }
}
