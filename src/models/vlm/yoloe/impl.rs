use anyhow::Result;
use ndarray::{s, Axis};
use rayon::prelude::*;
use std::collections::HashMap;

use crate::{
    elapsed_module, inputs,
    models::vision::{BoxType, YOLOPredsFormat, YOLO},
    Config, DynConf, Engine, Engines, FromConfig, Hbb, Image, ImageProcessor, Model, Module,
    NmsOps, Runtime, TextProcessor, Version, Xs, X, Y,
};

/// YOLOE with prompt-based inference (textual or visual prompts).
///
/// This model requires an embedding from either:
/// - `encode_class_names()` for textual prompts
/// - `encode_visual_prompt()` for visual prompts
#[derive(Debug)]
pub struct YOLOEPrompt {
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
}

impl Model for YOLOEPrompt {
    type Input<'a> = (&'a [Image], &'a X);

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

        if textual_encoder.is_some() && visual_encoder.is_some() {
            anyhow::bail!("YOLOEPrompt supports either visual or textual encoder, not both");
        }
        if textual_encoder.is_none() && visual_encoder.is_none() {
            anyhow::bail!(
                "YOLOEPrompt requires a textual or visual encoder. \
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

    fn run(
        &mut self,
        engines: &mut Engines,
        (images, embedding): Self::Input<'_>,
    ) -> Result<Vec<Y>> {
        let xs_images =
            elapsed_module!("YOLOEPrompt", "preprocess", self.processor.process(images)?);

        let ys = elapsed_module!("YOLOEPrompt", "inference", {
            let xs_images_x = xs_images.as_host()?;
            let embedding_view = if self.has_textual_encoder {
                let dim = embedding.dims()[1];
                embedding
                    .unsqueeze(0)?
                    .broadcast_to((images.len(), self.nc, dim))?
            } else {
                embedding.clone()
            };
            engines.run(
                &Module::Model,
                inputs![xs_images_x.view(), embedding_view.view()]?,
            )?
        });

        elapsed_module!("YOLOEPrompt", "postprocess", self.postprocess(&ys))
    }
}

impl YOLOEPrompt {
    /// Encode class names using textual encoder.
    pub fn encode_class_names(&mut self, engines: &mut Engines, class_names: &[&str]) -> Result<X> {
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

        let x = elapsed_module!("YOLOEPrompt", "textual-encoder-preprocess", {
            let texts: Vec<&str> = self.names.iter().map(|x| x.as_str()).collect();
            let encodings: Vec<f32> = text_processor
                .encode_texts_ids(&texts, true)?
                .into_iter()
                .flatten()
                .collect();
            let shape = &[texts.len(), encodings.len() / texts.len()];
            X::from_shape_vec(shape, encodings)?
        });

        let xs = elapsed_module!(
            "YOLOEPrompt",
            "textual-encoder-inference",
            engines.run(&Module::TextualEncoder, inputs![x]?)?
        );

        elapsed_module!("YOLOEPrompt", "textual-encoder-postprocess", {
            let x = xs
                .get::<f32>(0)
                .ok_or_else(|| anyhow::anyhow!("Failed to get textual encoder output"))?;
            let (n, dim) = (x.dims()[0], x.dims()[1]);
            let n_pad = self.nc.saturating_sub(n);
            if n_pad > 0 {
                let x_owned = x.to_owned();
                let x_zeros = X::zeros(&[n_pad, dim]);
                X::cat(&[x_owned, x_zeros], 0)
            } else {
                Ok(x.to_owned())
            }
        })
    }

    /// Encode visual prompt using visual encoder.
    pub fn encode_visual_prompt(
        &mut self,
        engines: &mut Engines,
        prompt_image: Image,
        hbbs: &[Hbb],
    ) -> Result<X> {
        let (image_embedding, mask, nc) =
            elapsed_module!("YOLOEPrompt", "visual-encoder-preprocess", {
                let image_embedding = self.processor.process(&[prompt_image])?.as_host()?;
                let ratio = self.processor.images_transform_info()[0].height_scale;

                let downsample_scale = 8.0;
                let scale_factor = ratio / downsample_scale;
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
                let mut class_groups: HashMap<
                    &str,
                    Vec<(usize, usize, usize, usize)>,
                > = HashMap::with_capacity(hbbs.len());

                self.names.clear();
                self.names.reserve(hbbs.len());

                for hbb in hbbs {
                    let class_name = hbb.name().unwrap_or("untitled");
                    let (x, y, w, h) = hbb.xywh();
                    let x1_f = (x * scale_factor).max(0.0).min(mask_w_f32 - 1.0);
                    let y1_f = (y * scale_factor).max(0.0).min(mask_h_f32 - 1.0);
                    let x2_f = ((x + w) * scale_factor).max(0.0).min(mask_w_f32);
                    let y2_f = ((y + h) * scale_factor).max(0.0).min(mask_h_f32);
                    let x2_f = x2_f.max(x1_f + min_size);
                    let y2_f = y2_f.max(y1_f + min_size);

                    let coords = (x1_f as usize, y1_f as usize, x2_f as usize, y2_f as usize);
                    class_groups.entry(class_name).or_default().push(coords);
                }

                let nc = class_groups.len();
                let mask_size = mask_h * mask_w;
                let mut mask_data = vec![0.0f32; nc * mask_size];

                let class_groups_vec: Vec<_> = class_groups.into_iter().collect();

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
            });

        self.nc = nc;

        elapsed_module!("YOLOEPrompt", "visual-encoder-inference", {
            let xs = engines.run(
                &Module::VisualEncoder,
                inputs![image_embedding.view(), mask.view()]?,
            )?;
            Ok(xs
                .get::<f32>(0)
                .ok_or_else(|| anyhow::anyhow!("Failed to get visual encoder output"))?
                .to_owned())
        })
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
            let (image_height, image_width, ratio) =
                (info.height_src, info.width_src, info.height_scale);

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

                // filter by conf
                if confidence < self.confs[class_id] {
                    return None;
                }

                // Bboxes
                let mut bbox_it = bbox.iter();
                let b0 = *bbox_it.next()? / ratio;
                let b1 = *bbox_it.next()? / ratio;
                let b2 = *bbox_it.next()? / ratio;
                let b3 = *bbox_it.next()? / ratio;
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
                        // let (cx, cy) = ((x + x2) / 2., (y + y2) / 2.);
                        (x, y, w, h)
                    }
                    BoxType::Xywh => {
                        let (x, y, w, h) = bbox;
                        // let (cx, cy) = (x + w / 2., y + h / 2.);
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
                if !self.names.is_empty() {
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

impl Runtime<YOLOEPrompt> {
    /// Encode class names using textual encoder.
    pub fn encode_class_names(&mut self, class_names: &[&str]) -> Result<X> {
        let (model, engines) = self.parts_mut();
        model.encode_class_names(engines, class_names)
    }

    /// Encode visual prompt using visual encoder.
    pub fn encode_visual_prompt(&mut self, prompt_image: Image, hbbs: &[Hbb]) -> Result<X> {
        let (model, engines) = self.parts_mut();
        model.encode_visual_prompt(engines, prompt_image, hbbs)
    }
}
