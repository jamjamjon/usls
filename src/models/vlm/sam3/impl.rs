use aksr::Builder;
use anyhow::Result;
use lru::LruCache;
use ndarray::{s, Array1, Array4};
use rayon::prelude::*;
use std::num::NonZeroUsize;

use crate::{
    elapsed_module, ort_inputs, vlm::Sam3Prompt, Config, DynConf, Engine, FromConfig, Hbb, Image,
    ImageProcessor, Mask, Ops, Task, TextProcessor, X, Y,
};

/// SAM3: Segment Anything with Concepts (Image) and SAM3 Tracker
#[derive(Builder, Debug)]
pub struct SAM3 {
    /// Task type: Sam3Image or Sam3Tracker
    task: Task,
    /// Vision encoder (shared for both tasks)
    visual_encoder: Engine,
    /// Text encoder (only for Sam3Image)
    textual_encoder: Option<Engine>,
    /// Geometry encoder (only for Sam3Image)
    geometry_encoder: Option<Engine>,
    /// Decoder (different for each task)
    decoder: Engine,
    vision_batch: usize,
    text_batch: usize,
    geo_batch: usize,
    decoder_batch: usize,
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
    conf: DynConf,
    text_cache: LruCache<String, (X, X)>,
    names: Vec<String>,
    spec: String,
}

impl SAM3 {
    pub fn new(mut config: Config) -> Result<Self> {
        let task = config.task.take().unwrap_or(Task::Sam3Image);

        // Take modules based on task type
        let visual_encoder =
            Engine::from_config(config.take_module(&crate::Module::VisualEncoder)?)?;
        let vision_batch = visual_encoder.batch().opt();
        let height = visual_encoder.try_height().unwrap_or(&1008.into()).opt();
        let width = visual_encoder.try_width().unwrap_or(&1008.into()).opt();
        let conf = DynConf::new_or_default(config.class_confs(), 1);

        let (
            textual_encoder,
            geometry_encoder,
            text_batch,
            geo_batch,
            decoder,
            decoder_batch,
            spec,
        ) = match task {
            Task::Sam3Image => {
                let te = Engine::from_config(config.take_module(&crate::Module::TextualEncoder)?)?;
                let ge = Engine::from_config(config.take_module(&crate::Module::Encoder)?)?;
                let decoder = Engine::from_config(config.take_module(&crate::Module::Decoder)?)?;
                let tb = te.batch().opt();
                let gb = ge.batch().opt();
                let db = decoder.batch().opt();
                (
                    Some(te),
                    Some(ge),
                    tb,
                    gb,
                    decoder,
                    db,
                    "sam3-image".to_string(),
                )
            }
            Task::Sam3Tracker => {
                let decoder = Engine::from_config(config.take_module(&crate::Module::Decoder)?)?;
                let db = decoder.batch().opt();
                (None, None, 1, 1, decoder, db, "sam3-tracker".to_string())
            }
            _ => anyhow::bail!("Unsupported task for SAM3: {:?}", task),
        };

        // Now consume config fields
        let image_processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);
        #[cfg(feature = "vlm")]
        let text_processor = TextProcessor::from_config(config.text_processor)?;
        #[cfg(not(feature = "vlm"))]
        let text_processor = TextProcessor::default();

        Ok(Self {
            task,
            visual_encoder,
            textual_encoder,
            geometry_encoder,
            decoder,
            image_processor,
            text_processor,
            conf,
            vision_batch,
            text_batch,
            geo_batch,
            decoder_batch,
            text_cache: LruCache::new(NonZeroUsize::new(100).unwrap()),
            names: vec![],
            spec,
        })
    }

    fn encode_images(&mut self, xs: &[Image]) -> Result<Vec<X>> {
        let xs_ = self.image_processor.process(xs)?;
        let output = self.visual_encoder.run(ort_inputs![xs_]?)?;
        let xs_out: Vec<X> = (0..output.len())
            .map(|i| X::from(output.get::<f32>(i).unwrap()))
            .collect();
        Ok(xs_out)
    }

    fn encode_texts(&mut self, texts: &[&str]) -> Result<Vec<(X, X)>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let textual_encoder = self
            .textual_encoder
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Text encoder not available for this task"))?;

        let mut all_results = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.text_batch) {
            let n = chunk.len();
            let encodings = self.text_processor.encode_texts(chunk, true)?;
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

            let ys = textual_encoder.run(ort_inputs![input_ids, attention_mask]?)?;
            let ys0 = X::from(ys.get::<f32>(0)?);
            let ys1 = X::from(ys.get::<f32>(1)?);

            for i in 0..n {
                all_results.push((
                    ys0.slice(s![i..i + 1, .., ..]).to_owned().into_dyn().into(),
                    ys1.slice(s![i..i + 1, ..]).to_owned().into_dyn().into(),
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

            let geometry_encoder = self
                .geometry_encoder
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("Geometry encoder not available for this task"))?;

            let ys = geometry_encoder.run(ort_inputs![
                input_boxes,
                input_labels,
                fpn_feat_batch,
                fpn_pos_batch
            ]?)?;

            // Split chunk results
            let geo_features = X::from(ys.get::<f32>(0)?);
            let geo_masks = X::from(ys.get::<f32>(1)?);

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
        fpn_features: &[X],
        prompt_features: &[X],
        prompt_masks: &[X],
    ) -> Result<Vec<Vec<X>>> {
        let n = prompt_features.len();
        if n == 0 {
            return Ok(vec![]);
        }

        // Concatenate prompts and broadcast FPN features
        let prompt_feat = X::concat(prompt_features, 0)?;
        let prompt_mask = X::concat(prompt_masks, 0)?;

        let (s0, s1, s2, s4) = (
            fpn_features[0].shape(),
            fpn_features[1].shape(),
            fpn_features[2].shape(),
            fpn_features[3].shape(),
        );

        let args = vec![
            fpn_features[0]
                .clone()
                .broadcast([n, s0[1], s0[2], s0[3]])?,
            fpn_features[1]
                .clone()
                .broadcast([n, s1[1], s1[2], s1[3]])?,
            fpn_features[2]
                .clone()
                .broadcast([n, s2[1], s2[2], s2[3]])?,
            fpn_features[3]
                .clone()
                .broadcast([n, s4[1], s4[2], s4[3]])?,
            prompt_feat,
            prompt_mask,
        ];
        let ys = self.decoder.run(&args)?;
        let ys0 = X::from(ys.get::<f32>(0)?);
        let ys1 = X::from(ys.get::<f32>(1)?);
        let ys2 = X::from(ys.get::<f32>(2)?);
        let ys3 = X::from(ys.get::<f32>(3)?);

        // Split batch results
        Ok((0..n)
            .map(|i| {
                vec![
                    ys0.slice(s![i..i + 1, .., .., ..])
                        .to_owned()
                        .into_dyn()
                        .into(),
                    ys1.slice(s![i..i + 1, .., ..]).to_owned().into_dyn().into(),
                    ys2.slice(s![i..i + 1, ..]).to_owned().into_dyn().into(),
                    ys3.slice(s![i..i + 1, ..]).to_owned().into_dyn().into(),
                ]
            })
            .collect())
    }

    /// Main forward method - dispatches to task-specific implementation
    pub fn forward(&mut self, xs: &[Image], prompts: &[Sam3Prompt]) -> Result<Vec<Y>> {
        match self.task {
            Task::Sam3Image => self.forward_image(xs, prompts),
            Task::Sam3Tracker => self.forward_tracker(xs, prompts),
            _ => anyhow::bail!("Unsupported task: {:?}", self.task),
        }
    }

    /// Forward for SAM3 Image task (text + geometry prompts)
    fn forward_image(&mut self, xs: &[Image], prompts: &[Sam3Prompt]) -> Result<Vec<Y>> {
        if xs.is_empty() || prompts.is_empty() {
            return Ok(vec![]);
        }

        // Update class names from prompts
        self.names = prompts.iter().map(|p| p.text.clone()).collect();

        // Step 1: Vision Encoding
        let all_fpn_features = elapsed_module!("SAM3", "vision-encoder", self.encode_images(xs)?);

        // Store original image dimensions
        let image_dims: Vec<(usize, usize)> = self
            .image_processor
            .images_transform_info()
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
            let fpn_features: Vec<X> = (0..4)
                .map(|i| {
                    let feat = &all_fpn_features[i];
                    feat.slice(ndarray::s![img_idx..img_idx + 1, .., .., ..])
                        .to_owned()
                        .into_dyn()
                        .into()
                })
                .collect();

            // Step 3.1: Prepare geometry data (single pass with unzip)
            let (prompts_boxes, prompts_labels, has_geometry_flags): (Vec<_>, Vec<_>, Vec<_>) =
                prompts
                    .iter()
                    .map(|p| {
                        if p.should_use_geometry() {
                            (
                                p.normalized_boxes(*image_width as f32, *image_height as f32),
                                p.box_labels(),
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
                    &fpn_features[3],
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
            let all_outputs: Vec<Vec<X>> = elapsed_module!("SAM3", "decoder", {
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

    /// Forward for SAM3 Tracker task (point + box prompts)
    ///
    /// ONNX Model shapes:
    /// - Vision encoder input: [B, 3, 1008, 1008]
    /// - Vision encoder outputs: embeddings[0]=[B,32,288,288], [1]=[B,64,144,144], [2]=[B,256,72,72]
    /// - Decoder inputs:
    ///   - input_points: [B, num_objects, num_points, 2]
    ///   - input_labels: [B, num_objects, num_points] (int64)
    ///   - input_boxes: [B, num_objects, 4]
    ///   - image_embeddings.0/1/2: from vision encoder
    /// - Decoder outputs:
    ///   - iou_scores: [B, num_objects, 3]
    ///   - pred_masks: [B, num_objects, 3, H, W]
    ///   - object_score_logits: [B, num_objects, 1]
    ///
    /// Note: Each box represents one object to segment. Points can be combined
    /// with a box to refine the segmentation of that object.
    fn forward_tracker(&mut self, xs: &[Image], prompts: &[Sam3Prompt]) -> Result<Vec<Y>> {
        if xs.is_empty() || prompts.is_empty() {
            return Ok(vec![]);
        }

        // Step 1: Vision Encoding
        let image_embeddings =
            elapsed_module!("SAM3-Tracker", "vision-encoder", self.encode_images(xs)?);

        // Store original image dimensions and compute scale factors
        let image_metas: Vec<_> = self
            .image_processor
            .images_transform_info()
            .iter()
            .map(|info| {
                let h_src = info.height_src as f32;
                let w_src = info.width_src as f32;
                let h_dst = info.height_dst as f32;
                let w_dst = info.width_dst as f32;
                (h_src as usize, w_src as usize, w_dst / w_src, h_dst / h_src)
            })
            .collect();

        let mut results = Vec::with_capacity(xs.len());

        // Process each image
        for (img_idx, (orig_h, orig_w, scale_x, scale_y)) in image_metas.iter().enumerate() {
            // Extract embeddings for this image
            let emb0: X = image_embeddings[0]
                .slice(s![img_idx..img_idx + 1, .., .., ..])
                .to_owned()
                .into_dyn()
                .into();
            let emb1: X = image_embeddings[1]
                .slice(s![img_idx..img_idx + 1, .., .., ..])
                .to_owned()
                .into_dyn()
                .into();
            let emb2: X = image_embeddings[2]
                .slice(s![img_idx..img_idx + 1, .., .., ..])
                .to_owned()
                .into_dyn()
                .into();

            let mut image_masks: Vec<Mask> = Vec::new();
            let mut object_idx = 0usize;

            for prompt in prompts.iter() {
                // Determine how to process this prompt:
                // - If has boxes: each box is a separate object
                // - If only points: all points belong to one object
                let scaled_boxes = prompt.scaled_boxes_xyxy(*scale_x, *scale_y);
                let scaled_points = prompt.scaled_points(*scale_x, *scale_y);
                let point_labels = prompt.point_labels();

                if scaled_boxes.is_empty() && scaled_points.is_empty() {
                    continue;
                }

                // Case 1: Only points, no boxes - single object with all points
                if scaled_boxes.is_empty() {
                    let masks = self.decode_tracker_single_object(
                        &scaled_points,
                        &point_labels,
                        None,
                        &emb0,
                        &emb1,
                        &emb2,
                    )?;

                    for (iou, obj_score, mask_data, mask_h, mask_w) in masks {
                        if let Some(mask) = self.create_tracker_mask(
                            &mask_data,
                            mask_w,
                            mask_h,
                            *orig_w,
                            *orig_h,
                            iou,
                            obj_score,
                            object_idx,
                            prompt.class_name(),
                        )? {
                            image_masks.push(mask);
                            object_idx += 1;
                        }
                    }
                }
                // Case 2: Has boxes - each box is a separate object
                else {
                    for (box_idx, box_coords) in scaled_boxes.iter().enumerate() {
                        // For now, points are shared with first box only if no other boxes
                        // In typical usage: one box per object, optional points for refinement
                        let obj_points = if scaled_boxes.len() == 1 {
                            scaled_points.clone()
                        } else {
                            vec![]
                        };
                        let obj_labels = if scaled_boxes.len() == 1 {
                            point_labels.clone()
                        } else {
                            vec![]
                        };

                        let masks = self.decode_tracker_single_object(
                            &obj_points,
                            &obj_labels,
                            Some(box_coords),
                            &emb0,
                            &emb1,
                            &emb2,
                        )?;

                        for (iou, obj_score, mask_data, mask_h, mask_w) in masks {
                            // Use box_idx for multi-box prompts
                            let name = if scaled_boxes.len() > 1 {
                                format!("{}_{}", prompt.class_name(), box_idx)
                            } else {
                                prompt.class_name().to_string()
                            };

                            if let Some(mask) = self.create_tracker_mask(
                                &mask_data, mask_w, mask_h, *orig_w, *orig_h, iou, obj_score,
                                object_idx, &name,
                            )? {
                                image_masks.push(mask);
                                object_idx += 1;
                            }
                        }
                    }
                }
            }

            results.push(Y::default().with_masks(&image_masks));
        }

        Ok(results)
    }

    /// Decode a single object for SAM3-Tracker
    #[allow(clippy::type_complexity)]
    fn decode_tracker_single_object(
        &mut self,
        points: &[[f32; 2]],
        labels: &[i64],
        box_coords: Option<&[f32; 4]>,
        emb0: &X,
        emb1: &X,
        emb2: &X,
    ) -> Result<Vec<(f32, f32, Vec<f32>, usize, usize)>> {
        // Prepare input_points and input_labels: [1, 1, num_points, 2]
        let (input_points, input_labels) = if !points.is_empty() {
            let num_pts = points.len();
            let pts_flat: Vec<f32> = points.iter().flat_map(|p| p.iter()).copied().collect();
            let pts_arr = Array4::from_shape_vec((1, 1, num_pts, 2), pts_flat)?;
            let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
            let labels_arr = ndarray::Array3::from_shape_vec((1, 1, num_pts), labels_f32)?;
            (X::from(pts_arr.into_dyn()), X::from(labels_arr.into_dyn()))
        } else {
            // No points - use dummy with label -1
            let pts_arr = Array4::<f32>::zeros((1, 1, 1, 2));
            let labels_arr = ndarray::Array3::from_elem((1, 1, 1), -1.0f32);
            (X::from(pts_arr.into_dyn()), X::from(labels_arr.into_dyn()))
        };

        // Prepare input_boxes: [1, num_boxes, 4]
        let input_boxes = if let Some(coords) = box_coords {
            let boxes_arr = ndarray::Array3::from_shape_vec((1, 1, 4), coords.to_vec())?;
            X::from(boxes_arr.into_dyn())
        } else {
            // No boxes - empty tensor [1, 0, 4]
            let boxes_arr = ndarray::Array3::<f32>::zeros((1, 0, 4));
            X::from(boxes_arr.into_dyn())
        };

        // Run decoder
        let args = vec![
            input_points,
            input_labels,
            input_boxes,
            emb0.clone(),
            emb1.clone(),
            emb2.clone(),
        ];
        let decoder_outputs = elapsed_module!("SAM3-Tracker", "decoder", self.decoder.run(&args)?);

        // Parse outputs: iou_scores[1,1,3], pred_masks[1,1,3,H,W], obj_scores[1,1,1]
        let iou_scores = X::from(decoder_outputs.get::<f32>(0)?);
        let pred_masks = X::from(decoder_outputs.get::<f32>(1)?);
        let obj_scores = X::from(decoder_outputs.get::<f32>(2)?);

        let iou = iou_scores.slice(s![0, 0, ..]).to_owned();
        let masks = pred_masks.slice(s![0, 0, .., .., ..]).to_owned();
        let obj_logit = obj_scores[[0, 0, 0]];
        let obj_score = 1.0 / (1.0 + (-obj_logit).exp());

        // Find best mask by IoU
        let best_idx = iou
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let best_iou = iou[best_idx];
        let best_mask = masks.slice(s![best_idx, .., ..]).to_owned();
        let (mask_h, mask_w) = best_mask.dim();

        // Apply sigmoid
        let mask_probs: Vec<f32> = best_mask
            .iter()
            .map(|&v| {
                let v_clipped = v.clamp(-50.0, 50.0);
                1.0 / (1.0 + (-v_clipped).exp())
            })
            .collect();

        Ok(vec![(best_iou, obj_score, mask_probs, mask_h, mask_w)])
    }

    /// Create a Mask from tracker output
    #[allow(clippy::too_many_arguments)]
    fn create_tracker_mask(
        &self,
        mask_probs: &[f32],
        mask_w: usize,
        mask_h: usize,
        orig_w: usize,
        orig_h: usize,
        iou: f32,
        obj_score: f32,
        object_idx: usize,
        name: &str,
    ) -> Result<Option<Mask>> {
        let luma = Ops::resize_lumaf32_u8(
            mask_probs,
            mask_w as _,
            mask_h as _,
            orig_w as _,
            orig_h as _,
            false,
            "Bilinear",
        )?;

        let luma_binary: Vec<u8> = luma
            .iter()
            .map(|&v| if v > 127 { 255 } else { 0 })
            .collect();

        if let Ok(mask) = Mask::new(&luma_binary, orig_w as u32, orig_h as u32) {
            Ok(Some(
                mask.with_id(object_idx)
                    .with_name(name)
                    .with_confidence(iou * obj_score),
            ))
        } else {
            Ok(None)
        }
    }

    fn postprocess(
        &self,
        outputs: &[X],
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
            .as_ref()
            .and_then(|e| e.odtypes())
            .and_then(|d| d.first().map(|t| t.size_in_bytes()))
            .unwrap_or(4);
        // Text encoder outputs: (1, 32, 256) + (1, 32)
        let bytes_per_entry = (32 * 256 + 32) * dtype_size;
        let mem = crate::human_bytes_binary((n * bytes_per_entry) as f64, 2);
        let texts: Vec<_> = self.text_cache.iter().map(|(k, _)| k.as_str()).collect();
        println!("Text Cache: {n}/{cap}, ~{mem} | {texts:?}");
    }

    /// Get the current task type
    pub fn current_task(&self) -> &Task {
        &self.task
    }
}
