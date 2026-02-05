use anyhow::Result;
use lru::LruCache;
use ndarray::{s, Axis};
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::{
    elapsed_module, inputs,
    models::vision::{BoxType, YOLOPredsFormat, YOLO},
    Config, DynConf, Engine, Engines, FromConfig, Hbb, Image, ImageProcessor, Model, Module,
    NmsOps, TextProcessor, Version, XView, Xs, YOLOEPrompt, X, Y,
};

/// YOLOE with prompt-based inference (textual or visual prompts).
#[derive(Debug)]
pub struct YOLOEPromptBased {
    pub height: usize,
    pub width: usize,
    pub batch: usize,
    pub spec: String,
    pub layout: YOLOPredsFormat,
    pub names: Vec<String>,
    pub nc: usize,
    pub confs: DynConf,
    pub iou: f32,
    pub processor: ImageProcessor,
    pub text_processor: Option<TextProcessor>,
    pub has_visual_encoder: bool,
    pub has_textual_encoder: bool,
    pub version: Option<Version>,
    textual_cache: LruCache<u64, Arc<X>>,
    visual_cache: LruCache<u64, (Arc<X>, Vec<String>)>,
}

impl Model for YOLOEPromptBased {
    type Input<'a> = (&'a [Image], &'a YOLOEPrompt);

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let textual_encoder = config
            .take_module(&Module::TextualEncoder)
            .ok()
            .map(Engine::from_config)
            .transpose()?;
        let visual_encoder = config
            .take_module(&Module::VisualEncoder)
            .ok()
            .map(Engine::from_config)
            .transpose()?;

        if textual_encoder.is_none() && visual_encoder.is_none() {
            anyhow::bail!(
                "YOLOEPromptBased requires a textual or visual encoder. \
                 For prompt-free models, use YOLOEPromptFree instead."
            );
        }

        let model_engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let (batch, height, width, spec) = (
            model_engine.batch().opt(),
            model_engine.try_height().unwrap_or(&640.into()).opt(),
            model_engine.try_width().unwrap_or(&640.into()).opt(),
            model_engine.spec().to_string(),
        );

        if let Some(ref ve) = visual_encoder {
            if ve.try_height().unwrap_or(&640.into()).opt() != height {
                anyhow::bail!(
                    "Visual encoder height mismatch: {} vs model {}",
                    ve.try_height().unwrap_or(&640.into()).opt(),
                    height
                );
            }
            if ve.try_width().unwrap_or(&640.into()).opt() != width {
                anyhow::bail!(
                    "Visual encoder width mismatch: {} vs model {}",
                    ve.try_width().unwrap_or(&640.into()).opt(),
                    width
                );
            }
        }

        let version = config.version;
        let layout = match version {
            Some(Version(8, _, _)) | Some(Version(11, _, _)) => {
                YOLOPredsFormat::n_cxcywh_clss_coefs_a()
            }
            Some(Version(26, _, _)) => YOLOPredsFormat::n_a_xyxy_confcls_coefs().apply_nms(false),
            Some(x) => anyhow::bail!("Unsupported YOLOE Version: {x:?}"),
            None => anyhow::bail!("No clear YOLOE Version specified."),
        };

        let names: Vec<String> = config.inference.class_names;
        let nc = config.inference.num_classes.unwrap_or(names.len()).max(1);
        let confs = DynConf::new_or_default(&config.inference.class_confs, nc);
        let iou = config.inference.iou.unwrap_or(0.45);
        let has_visual_encoder = visual_encoder.is_some();
        let has_textual_encoder = textual_encoder.is_some();
        let text_processor = if has_textual_encoder {
            Some(TextProcessor::from_config(config.text_processor)?)
        } else {
            None
        };
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            height,
            width,
            batch,
            spec,
            layout,
            names,
            nc,
            confs,
            iou,
            processor,
            text_processor,
            has_visual_encoder,
            has_textual_encoder,
            version,
            textual_cache: LruCache::new(NonZeroUsize::new(32).unwrap()),
            visual_cache: LruCache::new(NonZeroUsize::new(16).unwrap()),
        };

        let mut engines = Engines::new();
        engines.insert(Module::Model, model_engine);
        if let Some(te) = textual_encoder {
            engines.insert(Module::TextualEncoder, te);
        }
        if let Some(ve) = visual_encoder {
            engines.insert(Module::VisualEncoder, ve);
        }

        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, (images, prompt): Self::Input<'_>) -> Result<Vec<Y>> {
        // Determine prompt type and get/compute embedding
        let (embedding, nc) = if prompt.is_textual() {
            elapsed_module!("YOLOE-Prompt-Based", "encode-textual-prompt", {
                self.get_or_compute_textual_embedding(engines, prompt)?
            })
        } else if prompt.is_visual() {
            elapsed_module!("YOLOE-Prompt-Based", "encode-visual-prompt", {
                self.get_or_compute_visual_embedding(engines, prompt)?
            })
        } else {
            anyhow::bail!("YOLOEPrompt must be either textual or visual");
        };

        // Update nc for visual prompts (can vary per prompt)
        self.nc = nc;

        let xs_images = elapsed_module!(
            "YOLOE-Prompt-Based",
            "preprocess",
            self.processor.process(images)?
        );

        let ys = elapsed_module!("YOLOE-Prompt-Based", "inference", {
            let dims = embedding.dims();

            if prompt.is_textual() {
                // Textual embedding is typically [nc, dim] (2D)
                if dims.len() != 2 {
                    anyhow::bail!(
                        "Invalid textual embedding shape for YOLOE prompt inference: {dims:?}."
                    );
                }

                let dim = dims[1];
                if images.len() == 1 {
                    // Fast path: zero-copy reshape to [1, nc, dim]
                    let view_3d = embedding.0.view().insert_axis(Axis(0));
                    engines.run(&Module::Model, inputs![&xs_images, XView::from(view_3d)]?)?
                } else {
                    // Need a real contiguous [B, nc, dim] tensor
                    let embedding_broadcast =
                        embedding
                            .unsqueeze(0)?
                            .broadcast_to((images.len(), self.nc, dim))?;
                    engines.run(
                        &Module::Model,
                        inputs![&xs_images, embedding_broadcast.view()]?,
                    )?
                }
            } else {
                // Visual embedding is typically [1, nc, dim] (3D)
                if dims.len() != 3 {
                    anyhow::bail!(
                        "Invalid visual embedding shape for YOLOE prompt inference: {dims:?}."
                    );
                }

                if dims[0] == images.len() {
                    engines.run(&Module::Model, inputs![&xs_images, embedding.view()]?)?
                } else if dims[0] == 1 && images.len() > 1 {
                    let embedding_broadcast =
                        embedding.broadcast_to((images.len(), dims[1], dims[2]))?;
                    engines.run(
                        &Module::Model,
                        inputs![&xs_images, embedding_broadcast.view()]?,
                    )?
                } else if dims[0] == 1 {
                    engines.run(&Module::Model, inputs![&xs_images, embedding.view()]?)?
                } else {
                    anyhow::bail!(
                        "Invalid visual embedding batch dimension for YOLOE prompt inference: {dims:?}."
                    );
                }
            }
        });

        elapsed_module!("YOLOE-Prompt-Based", "postprocess", self.postprocess(&ys))
    }
}

impl YOLOEPromptBased {
    /// Generate cache key for textual prompts
    fn textual_cache_key(texts: &[String]) -> u64 {
        let mut hasher = DefaultHasher::new();
        texts.hash(&mut hasher);
        hasher.finish()
    }

    /// Generate cache key for visual prompts (based on box coordinates)
    fn visual_cache_key(image: &Image, boxes: &[Hbb]) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Image identity (best-effort): prefer source path if present, otherwise hash raw bytes.
        image.source.hash(&mut hasher);
        image.timestamp.map(|t| t.to_bits()).hash(&mut hasher);
        image.width().hash(&mut hasher);
        image.height().hash(&mut hasher);
        if image.source.is_none() {
            image.as_raw().hash(&mut hasher);
        }

        // Canonicalize boxes: order-independent hashing.
        let mut items: Vec<(&str, u32, u32, u32, u32)> = boxes
            .iter()
            .map(|hbb| {
                let name = hbb.name().unwrap_or("unnamed");
                let (x1, y1, x2, y2) = hbb.xyxy();
                (name, x1.to_bits(), y1.to_bits(), x2.to_bits(), y2.to_bits())
            })
            .collect();
        items.sort_unstable_by(|a, b| {
            a.0.cmp(b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.3.cmp(&b.3))
                .then_with(|| a.4.cmp(&b.4))
        });
        items.hash(&mut hasher);

        hasher.finish()
    }

    /// Get or compute textual embedding with caching.
    fn get_or_compute_textual_embedding(
        &mut self,
        engines: &mut Engines,
        prompt: &YOLOEPrompt,
    ) -> Result<(Arc<X>, usize)> {
        if !self.has_textual_encoder {
            anyhow::bail!(
                "Textual prompt provided but no textual encoder loaded. \
                 Use a config with textual encoder (e.g., yoloe_v8s_seg_tp)."
            );
        }

        let texts = match prompt {
            YOLOEPrompt::Textual(texts) => texts.as_slice(),
            _ => anyhow::bail!("Invalid textual prompt"),
        };
        if texts.is_empty() {
            anyhow::bail!("Invalid textual prompt: no class names");
        }

        let cache_key = Self::textual_cache_key(texts);

        // Check cache
        if let Some(cached) = self.textual_cache.get(&cache_key) {
            // Restore names from prompt
            self.names.clear();
            self.names.extend(texts.iter().cloned());
            return Ok((Arc::clone(cached), self.nc));
        }

        // Compute embedding
        let class_names: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embedding = self.encode_class_names_internal(engines, &class_names)?;

        let embedding = Arc::new(embedding);

        // Cache result
        self.textual_cache.put(cache_key, Arc::clone(&embedding));

        Ok((embedding, self.nc))
    }

    /// Get or compute visual embedding with caching.
    fn get_or_compute_visual_embedding(
        &mut self,
        engines: &mut Engines,
        prompt: &YOLOEPrompt,
    ) -> Result<(Arc<X>, usize)> {
        if !self.has_visual_encoder {
            anyhow::bail!(
                "Visual prompt provided but no visual encoder loaded. \
                 Use a config with visual encoder (e.g., yoloe_v8s_seg_vp)."
            );
        }

        let boxes = prompt
            .boxes()
            .ok_or_else(|| anyhow::anyhow!("Invalid visual prompt"))?;

        // Get image from prompt
        let image = prompt
            .image()
            .ok_or_else(|| anyhow::anyhow!("Visual prompt has no image"))?;

        let cache_key = Self::visual_cache_key(image, boxes);

        // Check cache
        if let Some((cached, names)) = self.visual_cache.get(&cache_key) {
            self.names.clear();
            self.names.extend(names.iter().cloned());
            return Ok((Arc::clone(cached), self.names.len()));
        }

        let (embedding, nc) = self.encode_visual_prompt_internal(engines, boxes, image)?;

        // Cache result (including names for stable class-id mapping)
        self.visual_cache
            .put(cache_key, (Arc::clone(&embedding), self.names.clone()));

        Ok((embedding, nc))
    }

    /// Internal: Encode class names using textual encoder.
    fn encode_class_names_internal(
        &mut self,
        engines: &mut Engines,
        class_names: &[&str],
    ) -> Result<X> {
        if class_names.len() > self.nc {
            anyhow::bail!(
                "The length of provided class names: {} exceeds the configured number of classes: {}.",
                class_names.len(),
                self.nc,
            );
        }

        self.names.clear();
        self.names.extend(class_names.iter().map(|x| x.to_string()));

        let text_processor = self.text_processor.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Text processor is not initialized (textual encoder missing?)")
        })?;

        let x = {
            let texts: Vec<&str> = self.names.iter().map(|x| x.as_str()).collect();
            let encodings: Vec<f32> = text_processor
                .encode_texts_ids(&texts, true)?
                .into_iter()
                .flatten()
                .collect();
            let shape = &[texts.len(), encodings.len() / texts.len()];
            X::from_shape_vec(shape, encodings)?
        };

        let xs = engines.run(&Module::TextualEncoder, inputs![x]?)?;

        let x = xs
            .get::<f32>(0)
            .ok_or_else(|| anyhow::anyhow!("Failed to get textual encoder output"))?;
        let x_owned = x.to_owned();
        let x_owned = match self.version {
            Some(Version(26, _, _)) => {
                let denom = x_owned.norm_l2_keepdim(-1)? + 1e-12;
                x_owned / denom
            }
            _ => x_owned,
        };

        let (n, dim) = (x_owned.dims()[0], x_owned.dims()[1]);
        let n_pad = self.nc.saturating_sub(n);
        if n_pad > 0 {
            let x_zeros = X::zeros(&[n_pad, dim]);
            X::cat(&[x_owned, x_zeros], 0)
        } else {
            Ok(x_owned)
        }
    }

    /// Internal: Encode visual prompt using visual encoder.
    /// Single image with multiple boxes - boxes can have duplicate class names
    fn encode_visual_prompt_internal(
        &mut self,
        engines: &mut Engines,
        hbbs: &[Hbb],
        image: &Image,
    ) -> Result<(Arc<X>, usize)> {
        if hbbs.is_empty() {
            anyhow::bail!("Visual prompt requires at least one bounding box");
        }

        let (image_embedding, mask, nc) = {
            let image_embedding = self.processor.process(std::slice::from_ref(image))?;
            let info = &self.processor.images_transform_info()[0];
            let resize_mode = match self
                .processor
                .resize_mode_type() {
                    Some(ResizeModeType::Letterbox) => ResizeModeType::Letterbox,
                    Some(ResizeModeType::FitAdaptive) => ResizeModeType::FitAdaptive,
                    Some(ResizeModeType::FitExact) => ResizeModeType::FitExact,
                    Some(x) => anyhow::bail!("Unsupported resize mode for YOLOEPromptBased: {x:?}. Supported: FitExact, FitAdaptive, Letterbox"),
                    _ => anyhow::bail!("No resize mode specified. Supported: FitExact, FitAdaptive, Letterbox"),
                };

            let downsample_scale = 8.0;
            let (prompt_height, prompt_width) = (
                self.height as f32 / downsample_scale,
                self.width as f32 / downsample_scale,
            );

            let mask_h = prompt_height as usize;
            let mask_w = prompt_width as usize;
            let mask_w_f32 = mask_w as f32;
            let mask_h_f32 = mask_h as f32;
            let min_size = 1.0;

            #[allow(clippy::type_complexity)]
            let mut class_groups: HashMap<&str, Vec<(usize, usize, usize, usize)>> =
                HashMap::with_capacity(hbbs.len());

            self.names.clear();
            self.names.reserve(hbbs.len());

            for hbb in hbbs {
                let class_name = hbb.name().unwrap_or("untitled");
                let (x, y, w, h) = hbb.xywh();

                // Transform box coordinates based on resize mode
                let (x1_transformed, y1_transformed, x2_transformed, y2_transformed) =
                    match resize_mode {
                        crate::ResizeModeType::Letterbox => {
                            let ratio = info.height_scale;
                            let pad_w = info.width_pad;
                            let pad_h = info.height_pad;
                            (
                                (x * ratio + pad_w) / downsample_scale,
                                (y * ratio + pad_h) / downsample_scale,
                                ((x + w) * ratio + pad_w) / downsample_scale,
                                ((y + h) * ratio + pad_h) / downsample_scale,
                            )
                        }
                        crate::ResizeModeType::FitAdaptive => {
                            let ratio = info.height_scale;
                            let scale_factor = ratio / downsample_scale;
                            (
                                x * scale_factor,
                                y * scale_factor,
                                (x + w) * scale_factor,
                                (y + h) * scale_factor,
                            )
                        }
                        crate::ResizeModeType::FitExact => {
                            let scale_x =
                                self.width as f32 / info.width_src as f32 / downsample_scale;
                            let scale_y =
                                self.height as f32 / info.height_src as f32 / downsample_scale;
                            (
                                x * scale_x,
                                y * scale_y,
                                (x + w) * scale_x,
                                (y + h) * scale_y,
                            )
                        }
                        _ => unreachable!(),
                    };

                let x1_f = x1_transformed.max(0.0).min(mask_w_f32 - 1.0);
                let y1_f = y1_transformed.max(0.0).min(mask_h_f32 - 1.0);
                let x2_f = x2_transformed.max(0.0).min(mask_w_f32);
                let y2_f = y2_transformed.max(0.0).min(mask_h_f32);
                let x2_f = x2_f.max(x1_f + min_size);
                let y2_f = y2_f.max(y1_f + min_size);

                let coords = (x1_f as usize, y1_f as usize, x2_f as usize, y2_f as usize);
                class_groups.entry(class_name).or_default().push(coords);
            }

            let nc = class_groups.len();
            let mask_size = mask_h * mask_w;
            let mut mask_data = vec![0.0f32; nc * mask_size];

            let mut class_groups_vec: Vec<_> = class_groups.into_iter().collect();
            class_groups_vec.sort_unstable_by(|a, b| a.0.cmp(b.0));

            for (class_name, _) in &class_groups_vec {
                self.names.push(class_name.to_string());
            }

            mask_data
                .par_chunks_mut(mask_size)
                .zip(class_groups_vec.par_iter())
                .for_each(|(mask_slice, (_class_name, boxes))| {
                    for &(x1, y1, x2, y2) in boxes {
                        for row in y1..y2 {
                            let row_start = row * mask_w + x1;
                            let row_end = row * mask_w + x2;
                            mask_slice[row_start..row_end].fill(1.0);
                        }
                    }
                });

            (
                image_embedding,
                X::from_shape_vec(&[1, nc, mask_h, mask_w], mask_data)?,
                nc,
            )
        };

        self.nc = nc;

        let embedding = {
            let xs = engines.run(
                &Module::VisualEncoder,
                inputs![&image_embedding, mask.view()]?,
            )?;
            xs.get::<f32>(0)
                .ok_or_else(|| anyhow::anyhow!("Failed to get visual encoder output"))?
                .to_owned()
        };

        Ok((Arc::new(embedding), nc))
    }

    fn postprocess(&self, outputs: &Xs) -> Result<Vec<Y>> {
        let preds = outputs
            .get::<f32>(0)
            .ok_or(anyhow::anyhow!("Failed to get the first output"))?;

        #[cfg(feature = "coreml")]
        let use_rayon_for_candidates = true;
        #[cfg(not(feature = "coreml"))]
        let use_rayon_for_candidates = !matches!(self.version.map(|v| v.0), Some(10 | 26));

        let f = |(idx, pred): (usize, _)| -> Option<Y> {
            let mut y = Y::default();

            // parse preds
            let (slice_bboxes, slice_id, slice_clss, slice_confs, _, slice_coefs, _) =
                self.layout.parse_preds(pred, self.nc);

            let info = &self.processor.images_transform_info[idx];
            let (image_height, image_width) = (info.height_src, info.width_src);
            let resize_mode = match self
                .processor
                .resize_mode_type() {
                    Some(ResizeModeType::Letterbox) => ResizeModeType::Letterbox,
                    Some(ResizeModeType::FitAdaptive) => ResizeModeType::FitAdaptive,
                    Some(ResizeModeType::FitExact) => ResizeModeType::FitExact,
                    Some(x) => anyhow::bail!("Unsupported resize mode for YOLOEPromptBased: {x:?}. Supported: FitExact, FitAdaptive, Letterbox"),
                    _ => anyhow::bail!("No resize mode specified. Supported: FitExact, FitAdaptive, Letterbox"),
                };

            // ObjectDetection
            let slice_bboxes = slice_bboxes?;
            let f_bbox = |(i, bbox): (usize, ndarray::ArrayViewD<'_, f32>)| -> Option<Hbb> {
                // confidence & class_id
                let (class_id, confidence) = match &slice_id {
                    Some(ids) => (ids[[i, 0]] as usize, slice_clss[[i, 0]]),
                    None => {
                        let (class_id, &confidence) =
                            slice_clss.slice(s![i, ..]).into_iter().enumerate().max_by(
                                |a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal),
                            )?;

                        match &slice_confs {
                            None => (class_id, confidence),
                            Some(slice_confs) => (class_id, confidence * slice_confs[[i, 0]]),
                        }
                    }
                };

                // For prompt-based inference, we may have fewer prompt names than `nc` due to padding.
                // Skip padded classes to avoid invalid indexing and spurious detections.
                if !self.names.is_empty() && class_id >= self.names.len() {
                    return None;
                }

                // filter by conf
                if confidence < self.confs[class_id] {
                    return None;
                }

                // Bboxes
                let mut bbox_it = bbox.iter();
                let (b0, b1, b2, b3) = (
                    *bbox_it.next()?,
                    *bbox_it.next()?,
                    *bbox_it.next()?,
                    *bbox_it.next()?,
                );

                let bbox = if self.layout.is_bbox_normalized {
                    (
                        b0 * self.width as f32,
                        b1 * self.height as f32,
                        b2 * self.width as f32,
                        b3 * self.height as f32,
                    )
                } else {
                    (b0, b1, b2, b3)
                };

                let bbox = match resize_mode {
                    crate::ResizeModeType::FitExact => {
                        let scale_x = image_width as f32 / self.width as f32;
                        let scale_y = image_height as f32 / self.height as f32;
                        (
                            bbox.0 * scale_x,
                            bbox.1 * scale_y,
                            bbox.2 * scale_x,
                            bbox.3 * scale_y,
                        )
                    }
                    crate::ResizeModeType::Letterbox => {
                        let ratio = info.height_scale;
                        let pad_w = info.width_pad;
                        let pad_h = info.height_pad;

                        match self.layout.box_type()? {
                            // All 4 values are positions
                            BoxType::Xyxy | BoxType::Cxcyxy | BoxType::XyCxcy => (
                                (bbox.0 - pad_w) / ratio,
                                (bbox.1 - pad_h) / ratio,
                                (bbox.2 - pad_w) / ratio,
                                (bbox.3 - pad_h) / ratio,
                            ),
                            // First two are positions, last two are sizes
                            BoxType::Cxcywh | BoxType::Xywh => (
                                (bbox.0 - pad_w) / ratio,
                                (bbox.1 - pad_h) / ratio,
                                bbox.2 / ratio,
                                bbox.3 / ratio,
                            ),
                        }
                    }
                    crate::ResizeModeType::FitAdaptive => {
                        let ratio = info.height_scale;
                        (
                            bbox.0 / ratio,
                            bbox.1 / ratio,
                            bbox.2 / ratio,
                            bbox.3 / ratio,
                        )
                    }
                    _ => unreachable!(),
                };

                let (x, y, w, h) = match self.layout.box_type()? {
                    BoxType::Cxcywh => {
                        let (cx, cy, w, h) = bbox;
                        let x = (cx - w / 2.).max(0.);
                        let y = (cy - h / 2.).max(0.);
                        (x, y, w, h)
                    }
                    BoxType::Xyxy => {
                        let (x, y, x2, y2) = bbox;
                        let (w, h) = (x2 - x, y2 - y);
                        (x, y, w, h)
                    }
                    BoxType::Xywh => {
                        let (x, y, w, h) = bbox;
                        (x, y, w, h)
                    }
                    BoxType::Cxcyxy => {
                        let (cx, cy, x2, y2) = bbox;
                        let (w, h) = ((x2 - cx) * 2., (y2 - cy) * 2.);
                        let x = (x2 - w).max(0.);
                        let y = (y2 - h).max(0.);
                        (x, y, w, h)
                    }
                    BoxType::XyCxcy => {
                        let (x, y, cx, cy) = bbox;
                        let (w, h) = ((cx - x) * 2., (cy - y) * 2.);
                        (x, y, w, h)
                    }
                };

                let mut hbb = Hbb::default()
                    .with_xywh(x, y, w, h)
                    .with_confidence(confidence)
                    .with_id(class_id)
                    .with_uid(i);
                if class_id < self.names.len() {
                    hbb = hbb.with_name(&self.names[class_id]);
                }

                Some(hbb)
            };

            let y_hbbs = if use_rayon_for_candidates {
                slice_bboxes
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .filter_map(f_bbox)
                    .collect::<Vec<_>>()
            } else {
                slice_bboxes
                    .axis_iter(Axis(0))
                    .enumerate()
                    .filter_map(f_bbox)
                    .collect::<Vec<_>>()
            };

            let mut y_hbbs = y_hbbs;

            // Bboxes
            if !y_hbbs.is_empty() {
                if self.layout.apply_nms {
                    y_hbbs.apply_nms_inplace(self.iou);
                }
                y = y.with_hbbs(&y_hbbs);
            }

            // InstanceSegmentation
            if let Some(coefs) = slice_coefs {
                if !y.hbbs().is_empty() {
                    let protos = outputs.get::<f32>(1);
                    if let Some(proto) = protos.as_ref() {
                        let proto_slice = proto.slice(s![idx, .., .., ..]);
                        let coefs_2d = coefs.into_dimensionality::<ndarray::Ix2>().ok()?;
                        let proto_3d = proto_slice.into_dimensionality::<ndarray::Ix3>().ok()?;
                        let y_masks = YOLO::generate_masks(
                            y.hbbs(),
                            coefs_2d,
                            proto_3d,
                            image_width,
                            image_height,
                            self.width,
                            self.height,
                            resize_mode,
                            info,
                        );
                        if !y_masks.is_empty() {
                            y = y.with_masks(&y_masks);
                        }
                    }
                }
            }

            Some(y)
        };

        let ys: Vec<Y> = if preds.len_of(Axis(0)) == 1 {
            preds.axis_iter(Axis(0)).enumerate().filter_map(f).collect()
        } else {
            preds
                .axis_iter(Axis(0))
                .into_par_iter()
                .enumerate()
                .filter_map(f)
                .collect()
        };

        Ok(ys)
    }
}
