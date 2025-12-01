use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Array1};
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};

use crate::{elapsed_module, Config, DynConf, Engine, Hbb, Image, Mask, Ops, Processor, Xs, X, Y};

/// SAM3 Prompt - ONE text + N boxes
///
/// Each prompt represents a single query:
/// - `text`: The text description (use "visual" for box-only mode)
/// - `boxes`: Bounding boxes in xywh format
/// - `labels`: Box labels (1=positive, 0=negative)
///
/// # Examples
/// ```
/// // Text only
/// Sam3Prompt::new("cat");
///
/// // Box only (defaults to "visual")
/// Sam3Prompt::new("visual")
///     .with_box(59.0, 144.0, 17.0, 19.0, true);
///
/// // Text + negative box
/// Sam3Prompt::new("handle")
///     .with_box(40.0, 183.0, 278.0, 21.0, false);
/// ```
#[derive(Debug, Clone)]
pub struct Sam3Prompt {
    /// Text prompt (single text per query)
    pub text: String,
    /// Bounding boxes in xywh format: [x, y, width, height]
    pub boxes: Vec<[f32; 4]>,
    /// Box labels: 1 = positive, 0 = negative
    pub labels: Vec<i64>,
}

impl Sam3Prompt {
    /// Default class name for box-only prompts
    pub const VISUAL: &'static str = "visual";
}

impl Default for Sam3Prompt {
    fn default() -> Self {
        Self {
            text: Self::VISUAL.to_string(),
            boxes: Vec::new(),
            labels: Vec::new(),
        }
    }
}

impl Sam3Prompt {
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            boxes: Vec::new(),
            labels: Vec::new(),
        }
    }

    /// Create a box-only prompt (text defaults to "visual")
    pub fn visual() -> Self {
        Self::new(Self::VISUAL)
    }

    /// Add a bounding box with label (xywh: x, y, width, height)
    pub fn with_box(mut self, x: f32, y: f32, w: f32, h: f32, positive: bool) -> Self {
        self.boxes.push([x, y, w, h]);
        self.labels.push(if positive { 1 } else { 0 });
        self
    }

    /// Add a positive bounding box
    pub fn with_positive_box(self, x: f32, y: f32, w: f32, h: f32) -> Self {
        self.with_box(x, y, w, h, true)
    }

    /// Add a negative bounding box
    pub fn with_negative_box(self, x: f32, y: f32, w: f32, h: f32) -> Self {
        self.with_box(x, y, w, h, false)
    }

    /// Check if prompt has any boxes
    pub fn has_boxes(&self) -> bool {
        !self.boxes.is_empty()
    }

    /// Check if prompt has at least one positive box
    pub fn has_positive_box(&self) -> bool {
        self.labels.iter().any(|&l| l == 1)
    }

    /// Check if this is a box-only prompt (text is "visual")
    pub fn is_visual(&self) -> bool {
        self.text == Self::VISUAL
    }

    /// Check if geometry should be used:
    /// - Box-only ("visual"): requires at least one positive box
    /// - Text + boxes: any boxes are valid (text provides anchor)
    pub fn should_use_geometry(&self) -> bool {
        if self.is_visual() {
            self.has_positive_box()
        } else {
            self.has_boxes()
        }
    }

    pub fn normalized_boxes(&self, image_width: f32, image_height: f32) -> Vec<[f32; 4]> {
        self.boxes
            .iter()
            .map(|&[x, y, w, h]| {
                let cx = (x + w / 2.0) / image_width;
                let cy = (y + h / 2.0) / image_height;
                let nw = w / image_width;
                let nh = h / image_height;
                [cx, cy, nw, nh]
            })
            .collect()
    }

    pub fn class_name(&self) -> &str {
        &self.text
    }

    fn parse_coords(s: &str) -> std::result::Result<[f32; 4], String> {
        let coords: Vec<f32> = s
            .split(',')
            .map(|x| x.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| format!("Invalid coordinates '{}': {}", s, e))?;

        if coords.len() != 4 {
            return Err(format!(
                "Expected 4 coordinates (x,y,w,h), got {}",
                coords.len()
            ));
        }

        Ok([coords[0], coords[1], coords[2], coords[3]])
    }
}

impl std::str::FromStr for Sam3Prompt {
    type Err = String;

    /// Parse from CLI format: "text;pos:x,y,w,h;neg:x,y,w,h"
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(';').collect();
        if parts.is_empty() {
            return Err("Empty prompt string".to_string());
        }

        let text = parts[0].trim();
        let mut prompt = Self::new(text);

        for part in parts.iter().skip(1) {
            let part = part.trim();
            if let Some(coords) = part.strip_prefix("pos:") {
                let [x, y, w, h] = Self::parse_coords(coords)?;
                prompt = prompt.with_positive_box(x, y, w, h);
            } else if let Some(coords) = part.strip_prefix("neg:") {
                let [x, y, w, h] = Self::parse_coords(coords)?;
                prompt = prompt.with_negative_box(x, y, w, h);
            } else {
                return Err(format!(
                    "Invalid box format: '{}'. Use 'pos:x,y,w,h' or 'neg:x,y,w,h'",
                    part
                ));
            }
        }

        Ok(prompt)
    }
}

/// SAM3 - Segment Anything Model 3
#[derive(Builder, Debug)]
pub struct SAM3 {
    visual_encoder: Engine,
    textual_encoder: Engine,
    geometry_encoder: Engine,
    decoder: Engine,
    processor: Processor,
    conf: DynConf,
    /// Image batch size (for vision encoder)
    batch: usize,
    /// Text batch size (for text encoder)
    text_batch: usize,
    /// Maximum cache size (number of texts)
    #[args(skip)]
    text_cache_max_size: usize,
    /// Enable prompt batching for geometry encoder and decoder
    #[args(skip)]
    enable_prompt_batching: bool,
    /// Text feature cache: text -> (features, mask)
    #[args(skip)]
    text_feature_cache: HashMap<String, (X, X)>,
    /// LRU tracking: least recently used texts
    #[args(skip)]
    text_cache_lru: VecDeque<String>,
    names: Vec<String>,
    spec: String,
}

impl SAM3 {
    pub fn new(config: Config) -> Result<Self> {
        let visual_encoder = Engine::try_from_config(&config.visual_encoder)?;
        let textual_encoder = Engine::try_from_config(&config.textual_encoder)?;
        let geometry_encoder = Engine::try_from_config(&config.encoder)?;
        let decoder = Engine::try_from_config(&config.decoder)?;
        let batch = visual_encoder.batch().opt();
        let text_batch = textual_encoder.batch().opt();
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
            batch,
            text_batch,
            text_cache_max_size: 100, // Default: cache up to 100 unique texts
            enable_prompt_batching: false, // Default: disabled (broadcast cost > benefit)
            text_feature_cache: HashMap::new(),
            text_cache_lru: VecDeque::new(),
            names: vec![],
            spec: "sam3".to_string(),
        })
    }

    /// Configure text cache max size
    pub fn with_cache_size(mut self, max_size: usize) -> Self {
        self.text_cache_max_size = max_size;
        self
    }

    /// Enable/disable prompt batching (geometry + decoder)
    pub fn with_prompt_batching_enabled(mut self, enable: bool) -> Self {
        self.enable_prompt_batching = enable;
        self
    }

    /// Clear text feature cache
    pub fn clear_text_cache(&mut self) {
        self.text_feature_cache.clear();
        self.text_cache_lru.clear();
    }

    /// Get cache statistics: (cached_texts_count, estimated_memory_mb)
    pub fn cache_stats(&self) -> (usize, usize) {
        let count = self.text_feature_cache.len();
        // Rough estimation: each text feature ~ 256 seq_len * 256 dims * 4 bytes * 2 (feat + mask)
        let estimated_mb = count * 256 * 256 * 4 * 2 / 1024 / 1024;
        (count, estimated_mb)
    }

    /// Update LRU on cache access (move to back = most recently used)
    fn touch_cache_entry(&mut self, text: &str) {
        // Remove old position if exists
        if let Some(pos) = self.text_cache_lru.iter().position(|t| t == text) {
            self.text_cache_lru.remove(pos);
        }
        // Add to back (most recently used)
        self.text_cache_lru.push_back(text.to_string());
    }

    /// Evict least recently used entries if cache is full
    fn evict_lru_if_needed(&mut self) {
        while self.text_feature_cache.len() >= self.text_cache_max_size {
            if let Some(lru_text) = self.text_cache_lru.pop_front() {
                self.text_feature_cache.remove(&lru_text);
            } else {
                break;
            }
        }
    }

    /// Image encoding - extract FPN features
    fn encode_image(&mut self, xs: &[Image]) -> Result<Xs> {
        let xs_ = elapsed_module!("SAM3", "visual-preprocess", {
            self.processor.process_images(xs)?
        });
        elapsed_module!("SAM3", "visual-inference", {
            self.visual_encoder.run(Xs::from(xs_))
        })
    }

    fn encode_texts_batch(&mut self, texts: &[&str]) -> Result<Vec<(X, X)>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Split texts into chunks based on text_batch size
        let mut all_results = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.text_batch) {
            let batch_size = chunk.len();

            // Tokenize with automatic padding/truncation (add_special_tokens=true)
            let encodings = self.processor.encode_texts(chunk, true)?;

            // Build batched input_ids tensor
            let seq_len = encodings[0].get_ids().len();
            let all_ids: Vec<f32> = encodings
                .iter()
                .flat_map(|e| e.get_ids().iter().map(|&id| id as f32))
                .collect();

            // Build attention_mask: 1 for real tokens, 0 for padding
            let all_masks: Vec<f32> = encodings
                .iter()
                .flat_map(|e| e.get_attention_mask().iter().map(|&m| m as f32))
                .collect();

            let input_ids = X::from_shape_vec(&[batch_size, seq_len], all_ids)?;
            let attention_mask = X::from_shape_vec(&[batch_size, seq_len], all_masks)?;

            // Run batched text encoder
            let ys = elapsed_module!("SAM3", "text-encoder-chunk", {
                self.textual_encoder
                    .run(Xs::from(vec![input_ids, attention_mask]))?
            });

            // Split batch results: text_features [batch, seq, 256], text_mask [batch, seq]
            let text_features = &ys[0];
            let text_mask = &ys[1];

            // Extract individual results
            let chunk_results: Vec<(X, X)> = (0..batch_size)
                .map(|i| {
                    let feat = text_features
                        .slice(ndarray::s![i..i + 1, .., ..])
                        .to_owned()
                        .into_dyn()
                        .into();
                    let mask = text_mask
                        .slice(ndarray::s![i..i + 1, ..])
                        .to_owned()
                        .into_dyn()
                        .into();
                    (feat, mask)
                })
                .collect();

            all_results.extend(chunk_results);
        }

        Ok(all_results)
    }

    /// Geometry encoding - encode bounding box prompts (single batch)
    fn encode_geometry(
        &mut self,
        boxes: &[[f32; 4]],
        labels: &[i64],
        fpn_feat: &X,
        fpn_pos: &X,
    ) -> Result<(X, X)> {
        let num_boxes = boxes.len();
        let boxes_flat: Vec<f32> = boxes.iter().flat_map(|b| b.iter().copied()).collect();

        let input_boxes = X::from_shape_vec(&[1, num_boxes, 4], boxes_flat)?;
        // Convert i64 labels to f32 for X type (ONNX runtime handles type conversion)
        let labels_f32: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
        let input_labels = X::from_shape_vec(&[1, num_boxes], labels_f32)?;

        let ys = elapsed_module!("SAM3", "geometry-inference", {
            self.geometry_encoder.run(Xs::from(vec![
                input_boxes,
                input_labels,
                fpn_feat.clone(),
                fpn_pos.clone(),
            ]))?
        });

        Ok((ys[0].clone(), ys[1].clone()))
    }

    /// Batch geometry encoding - encode multiple prompts at once
    /// Returns Vec of (geometry_features, geometry_mask) for each prompt
    fn encode_geometry_batch(
        &mut self,
        prompts_boxes: &[Vec<[f32; 4]>],
        prompts_labels: &[Vec<i64>],
        fpn_feat: &X,
        fpn_pos: &X,
    ) -> Result<Vec<(X, X)>> {
        let batch_size = prompts_boxes.len();

        // Find max number of boxes across all prompts
        let max_boxes = prompts_boxes.iter().map(|b| b.len()).max().unwrap_or(0);
        if max_boxes == 0 {
            return Ok(vec![
                (X::zeros(&[1, 1, 256]), X::zeros(&[1, 1]));
                batch_size
            ]);
        }

        // Pad all prompts to same number of boxes
        let mut boxes_flat = Vec::with_capacity(batch_size * max_boxes * 4);
        let mut labels_flat = Vec::with_capacity(batch_size * max_boxes);

        for i in 0..batch_size {
            let boxes = &prompts_boxes[i];
            let labels = &prompts_labels[i];

            // Add actual boxes
            for box_ in boxes {
                boxes_flat.extend_from_slice(box_);
            }
            // Pad with zeros
            for _ in boxes.len()..max_boxes {
                boxes_flat.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
            }

            // Add actual labels
            labels_flat.extend(labels.iter().map(|&l| l as f32));
            // Pad with zeros
            let padding_count = max_boxes - labels.len();
            labels_flat.resize(labels_flat.len() + padding_count, 0.0);
        }

        let input_boxes = X::from_shape_vec(&[batch_size, max_boxes, 4], boxes_flat)?;
        let input_labels = X::from_shape_vec(&[batch_size, max_boxes], labels_flat)?;

        // Expand fpn features to batch dimension
        let fpn_feat_batch = fpn_feat.clone().broadcast([batch_size, 256, 72, 72])?;
        let fpn_pos_batch = fpn_pos.clone().broadcast([batch_size, 256, 72, 72])?;

        let ys = elapsed_module!("SAM3", "geometry-batch-inference", {
            self.geometry_encoder.run(Xs::from(vec![
                input_boxes,
                input_labels,
                fpn_feat_batch,
                fpn_pos_batch,
            ]))?
        });

        // Split batch results
        let geo_features = &ys[0]; // [batch, num_boxes+1, 256]
        let geo_masks = &ys[1]; // [batch, num_boxes+1]

        let results: Vec<(X, X)> = (0..batch_size)
            .map(|i| {
                let num_boxes = prompts_boxes[i].len();
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
                (feat, mask)
            })
            .collect();

        Ok(results)
    }

    /// Decoder - generate segmentation results (single prompt)
    fn decode(&mut self, fpn_features: &Xs, prompt_features: &X, prompt_mask: &X) -> Result<Xs> {
        elapsed_module!("SAM3", "decode-inference", {
            self.decoder.run(Xs::from(vec![
                fpn_features[0].clone(), // fpn_feat_0
                fpn_features[1].clone(), // fpn_feat_1
                fpn_features[2].clone(), // fpn_feat_2
                fpn_features[6].clone(), // fpn_pos_2
                prompt_features.clone(),
                prompt_mask.clone(),
            ]))
        })
    }

    /// Batch decoder - decode multiple prompts at once
    fn decode_batch(
        &mut self,
        fpn_features: &Xs,
        prompt_features_batch: &[X],
        prompt_masks_batch: &[X],
    ) -> Result<Vec<Xs>> {
        let batch_size = prompt_features_batch.len();

        // Concatenate all prompt features along batch dimension
        let all_prompt_features = X::concat(prompt_features_batch, 0)?;
        let all_prompt_masks = X::concat(prompt_masks_batch, 0)?;

        // Broadcast FPN features to match batch size (dynamically get spatial dims)
        let s0 = fpn_features[0].shape();
        let s1 = fpn_features[1].shape();
        let s2 = fpn_features[2].shape();
        let s6 = fpn_features[6].shape();

        let fpn_feat_0_batch = fpn_features[0]
            .clone()
            .broadcast([batch_size, s0[1], s0[2], s0[3]])?;
        let fpn_feat_1_batch = fpn_features[1]
            .clone()
            .broadcast([batch_size, s1[1], s1[2], s1[3]])?;
        let fpn_feat_2_batch = fpn_features[2]
            .clone()
            .broadcast([batch_size, s2[1], s2[2], s2[3]])?;
        let fpn_pos_2_batch = fpn_features[6]
            .clone()
            .broadcast([batch_size, s6[1], s6[2], s6[3]])?;

        let ys = elapsed_module!("SAM3", "decode-batch-inference", {
            self.decoder.run(Xs::from(vec![
                fpn_feat_0_batch,
                fpn_feat_1_batch,
                fpn_feat_2_batch,
                fpn_pos_2_batch,
                all_prompt_features,
                all_prompt_masks,
            ]))?
        });

        // Split batch results: pred_masks, pred_boxes, pred_logits, presence_logits
        let pred_masks = &ys[0]; // [batch, 200, 288, 288]
        let pred_boxes = &ys[1]; // [batch, 200, 4]
        let pred_logits = &ys[2]; // [batch, 200]
        let presence_logits = &ys[3]; // [batch, 1]

        let results: Vec<Xs> = (0..batch_size)
            .map(|i| {
                vec![
                    pred_masks
                        .slice(ndarray::s![i..i + 1, .., .., ..])
                        .to_owned()
                        .into_dyn()
                        .into(),
                    pred_boxes
                        .slice(ndarray::s![i..i + 1, .., ..])
                        .to_owned()
                        .into_dyn()
                        .into(),
                    pred_logits
                        .slice(ndarray::s![i..i + 1, ..])
                        .to_owned()
                        .into_dyn()
                        .into(),
                    presence_logits
                        .slice(ndarray::s![i..i + 1, ..])
                        .to_owned()
                        .into_dyn()
                        .into(),
                ]
                .into()
            })
            .collect();

        Ok(results)
    }

    pub fn forward(&mut self, xs: &[Image], prompts: &[Sam3Prompt]) -> Result<Vec<Y>> {
        if xs.is_empty() || prompts.is_empty() {
            return Ok(vec![]);
        }

        // Update batch size based on actual input
        self.batch = xs.len();

        // Update class names from prompts
        self.names = prompts.iter().map(|p| p.text.clone()).collect();

        // Step 1: Vision Encoding (batch=N images)
        let all_fpn_features = elapsed_module!("SAM3", "vision-encoder", self.encode_image(xs)?);

        // Store original image dimensions
        let image_dims: Vec<(usize, usize)> = self
            .processor
            .images_transform_info
            .iter()
            .map(|info| (info.height_src as usize, info.width_src as usize))
            .collect();

        // Step 2: Text Encoding (with smart caching)
        // 2.1 Find unique texts that need encoding
        let unique_texts: std::collections::HashSet<&str> =
            prompts.iter().map(|p| p.text.as_str()).collect();

        let uncached_texts: Vec<&str> = unique_texts
            .iter()
            .filter(|&&t| !self.text_feature_cache.contains_key(t))
            .copied()
            .collect();

        // 2.2 Batch encode only uncached texts
        if !uncached_texts.is_empty() {
            let new_features = elapsed_module!(
                "SAM3",
                "text-encoder-cache-miss",
                self.encode_texts_batch(&uncached_texts)?
            );

            // 2.3 Update cache with LRU eviction
            for (text, feat) in uncached_texts.iter().zip(new_features.into_iter()) {
                // Evict LRU entries if cache is full
                self.evict_lru_if_needed();

                // Insert new entry
                self.text_feature_cache.insert(text.to_string(), feat);
                self.touch_cache_entry(text);
            }
        }

        // 2.4 Retrieve cached text features for all prompts (update LRU on access)
        let text_features_batch: Vec<(X, X)> = prompts
            .iter()
            .map(|p| {
                let text = p.text.as_str();
                self.touch_cache_entry(text);
                self.text_feature_cache[text].clone()
            })
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

            let mut combined_masks: Vec<Mask> = Vec::new();
            let mut combined_hbbs: Vec<Hbb> = Vec::new();

            // Choose between batch processing or sequential processing
            //
            // NOTE: Prompt batching is DISABLED by default because:
            // - Broadcast cost: FPN features × N prompts = ~200MB+ memory copy
            // - Marginal benefit: Kernel launch savings < memory bandwidth cost
            // - Better alternative: Use rayon for CPU parallelism (no GPU memory overhead)
            //
            // To enable: model.with_prompt_batching_enabled(true)
            if self.enable_prompt_batching && prompts.len() > 1 {
                // Batch Mode: Process all prompts together (GPU batching)

                // Prepare batch geometry encoding
                let mut prompts_boxes = Vec::new();
                let mut prompts_labels = Vec::new();
                let mut has_geometry_flags = Vec::new();

                for prompt in prompts.iter() {
                    if prompt.should_use_geometry() {
                        let norm_boxes =
                            prompt.normalized_boxes(*image_width as f32, *image_height as f32);
                        prompts_boxes.push(norm_boxes);
                        prompts_labels.push(prompt.labels.clone());
                        has_geometry_flags.push(true);
                    } else {
                        prompts_boxes.push(vec![]);
                        prompts_labels.push(vec![]);
                        has_geometry_flags.push(false);
                    }
                }

                // Batch encode geometry
                let geo_features_batch = self.encode_geometry_batch(
                    &prompts_boxes,
                    &prompts_labels,
                    &fpn_features[2],
                    &fpn_features[6],
                )?;

                // Build prompt features
                let mut prompt_features_list = Vec::new();
                let mut prompt_masks_list = Vec::new();

                for (idx, (has_geo, (text_features, text_mask))) in has_geometry_flags
                    .iter()
                    .zip(text_features_batch.iter())
                    .enumerate()
                {
                    let (prompt_features, prompt_mask) = if *has_geo {
                        let (geo_features, geo_mask) = geo_features_batch[idx].clone();
                        (
                            X::concat(&[text_features.clone(), geo_features], 1)?,
                            X::concat(&[text_mask.clone(), geo_mask], 1)?,
                        )
                    } else {
                        (text_features.clone(), text_mask.clone())
                    };
                    prompt_features_list.push(prompt_features);
                    prompt_masks_list.push(prompt_mask);
                }

                // Batch decode
                let outputs_batch =
                    self.decode_batch(&fpn_features, &prompt_features_list, &prompt_masks_list)?;

                // Post-process each result
                for (prompt_idx, (prompt, outputs)) in
                    prompts.iter().zip(outputs_batch.iter()).enumerate()
                {
                    let y = elapsed_module!(
                        "SAM3",
                        "postprocess",
                        self.postprocess(
                            outputs,
                            *image_height,
                            *image_width,
                            prompt_idx,
                            prompt.class_name()
                        )?
                    );
                    combined_masks.extend(y.masks().iter().cloned());
                    combined_hbbs.extend(y.hbbs().iter().cloned());
                }
            } else {
                // Sequential Mode: Process prompts one by one (Recommended)
                // FUTURE OPTIMIZATION: Use rayon for parallel processing
                // - No GPU memory overhead (each thread uses same FPN features)
                // - Better CPU utilization during I/O waits
                // - Easy to implement:
                //   ```rust
                //   use rayon::prelude::*;
                //   let results: Vec<_> = prompts.par_iter().enumerate().map(|(idx, prompt)| {
                //       // Process each prompt independently
                //   }).collect();
                //   ```
                // - NOTE: Requires thread-safe access to ONNX sessions (check Engine impl)
                //
                for (prompt_idx, prompt) in prompts.iter().enumerate() {
                    let (text_features, text_mask) = text_features_batch[prompt_idx].clone();

                    // Build prompt features based on geometry availability
                    let (prompt_features, prompt_mask) = if prompt.should_use_geometry() {
                        // Encode geometry (boxes + labels)
                        let norm_boxes =
                            prompt.normalized_boxes(*image_width as f32, *image_height as f32);
                        let (geo_features, geo_mask) = self.encode_geometry(
                            &norm_boxes,
                            &prompt.labels,
                            &fpn_features[2],
                            &fpn_features[6],
                        )?;

                        // Concatenate text and geometry features
                        let prompt_features = X::concat(&[text_features, geo_features], 1)?;
                        let prompt_mask = X::concat(&[text_mask, geo_mask], 1)?;
                        (prompt_features, prompt_mask)
                    } else {
                        // Text-only: use text features directly
                        (text_features, text_mask)
                    };

                    // Decode
                    let outputs = self.decode(&fpn_features, &prompt_features, &prompt_mask)?;

                    // Post-process
                    let y = elapsed_module!(
                        "SAM3",
                        "postprocess",
                        self.postprocess(
                            &outputs,
                            *image_height,
                            *image_width,
                            prompt_idx,
                            prompt.class_name()
                        )?
                    );

                    combined_masks.extend(y.masks().iter().cloned());
                    combined_hbbs.extend(y.hbbs().iter().cloned());
                }
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
}
