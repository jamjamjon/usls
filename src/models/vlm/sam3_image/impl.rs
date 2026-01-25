use aksr::Builder;
use anyhow::{anyhow, bail, Result};
use half::f16;
use lru::LruCache;
use ndarray::{s, Array1, ArrayD, Axis};
use ort::value::{DynValue, Value};
use rayon::prelude::*;
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::{
    elapsed_module, inputs, Config, DynConf, Engine, Engines, FromConfig, Hbb, Image,
    ImageProcessor, Mask, Model, Module, Ops, Sam3Prompt, TextProcessor, Xs, X, Y,
};

#[derive(Builder, Debug)]
pub struct Sam3Image {
    pub vision_batch: usize,
    pub text_batch: usize,
    pub decoder_batch: usize,
    pub image_processor: ImageProcessor,
    pub text_processor: TextProcessor,
    pub conf: DynConf,
    pub text_cache: LruCache<String, (Arc<DynValue>, Arc<DynValue>)>,
    pub names: Vec<String>,
    pub spec: String,
}

impl Sam3Image {
    fn extract_f32(val: &DynValue) -> Result<ArrayD<f32>> {
        use ort::tensor::TensorElementType as TE;
        use ort::value::ValueType;
        match val.dtype() {
            ValueType::Tensor { ty, .. } => match ty {
                TE::Float32 => Ok(val.try_extract_array::<f32>()?.into_owned()),
                TE::Float16 => Ok(val
                    .try_extract_array::<f16>()?
                    .mapv(|x| x.to_f32())
                    .into_owned()),
                TE::Bool => Ok(val
                    .try_extract_array::<bool>()?
                    .mapv(|x| if x { 1. } else { 0. })
                    .into_owned()),
                _ => bail!("Unsupported tensor type: {:?}", ty),
            },
            _ => bail!("Unsupported value type"),
        }
    }

    fn repeat_text_feat_to_batch(text_feat: &DynValue, batch: usize) -> Result<DynValue> {
        if batch <= 1 {
            anyhow::bail!("repeat_text_feat_to_batch requires batch > 1");
        }

        use ort::memory::AllocationDevice;
        use ort::tensor::TensorElementType as TE;
        use ort::value::ValueType;

        let owned = text_feat
            .view()
            .try_upgrade()
            .map_err(|_| anyhow!("Upgrade failed"))?;
        let tensor = owned
            .downcast::<ort::value::DynTensorValueType>()
            .map_err(|_| anyhow!("Failed to downcast to tensor"))?;
        let tensor = if tensor.memory_info().is_cpu_accessible() {
            tensor
        } else {
            tensor
                .to(AllocationDevice::CPU, 0)
                .or_else(|_| tensor.to(AllocationDevice::CUDA_PINNED, 0))?
        };
        let cpu_val = tensor.into_dyn();

        match cpu_val.dtype() {
            ValueType::Tensor { ty, .. } => match ty {
                TE::Float16 => {
                    let arr = cpu_val.try_extract_array::<f16>()?.into_owned().into_dyn();
                    let shape = arr.shape().to_vec();
                    if shape.is_empty() || shape[0] != 1 {
                        bail!("Expected text_features shape [1, ...], got {:?}", shape);
                    }
                    let per = arr.len();
                    let src = arr.into_raw_vec_and_offset().0;
                    let mut out = Vec::with_capacity(batch * per);
                    for _ in 0..batch {
                        out.extend_from_slice(&src);
                    }
                    let mut new_shape = shape;
                    new_shape[0] = batch;
                    Ok(Value::from_array(ndarray::Array::from_shape_vec(
                        ndarray::IxDyn(&new_shape),
                        out,
                    )?)?
                    .into_dyn())
                }
                TE::Float32 => {
                    let arr = cpu_val.try_extract_array::<f32>()?.into_owned().into_dyn();
                    let shape = arr.shape().to_vec();
                    if shape.is_empty() || shape[0] != 1 {
                        bail!("Expected text_features shape [1, ...], got {:?}", shape);
                    }
                    let per = arr.len();
                    let src = arr.into_raw_vec_and_offset().0;
                    let mut out = Vec::with_capacity(batch * per);
                    for _ in 0..batch {
                        out.extend_from_slice(&src);
                    }
                    let mut new_shape = shape;
                    new_shape[0] = batch;
                    Ok(Value::from_array(ndarray::Array::from_shape_vec(
                        ndarray::IxDyn(&new_shape),
                        out,
                    )?)?
                    .into_dyn())
                }
                _ => bail!("Unsupported text_features tensor type: {:?}", ty),
            },
            _ => bail!("Unsupported text_features value type"),
        }
    }

    fn repeat_text_mask_to_batch(text_mask: &DynValue, batch: usize) -> Result<DynValue> {
        if batch <= 1 {
            anyhow::bail!("repeat_text_mask_to_batch requires batch > 1");
        }

        use ort::memory::AllocationDevice;

        let owned = text_mask
            .view()
            .try_upgrade()
            .map_err(|_| anyhow!("Upgrade failed"))?;
        let tensor = owned
            .downcast::<ort::value::DynTensorValueType>()
            .map_err(|_| anyhow!("Failed to downcast to tensor"))?;
        let tensor = if tensor.memory_info().is_cpu_accessible() {
            tensor
        } else {
            tensor
                .to(AllocationDevice::CPU, 0)
                .or_else(|_| tensor.to(AllocationDevice::CUDA_PINNED, 0))?
        };
        let cpu_val = tensor.into_dyn();

        let arr = Self::extract_f32(&cpu_val)?.mapv(|x| x != 0.0);
        let shape = arr.shape().to_vec();
        if shape.is_empty() || shape[0] != 1 {
            bail!("Expected text_mask shape [1, ...], got {:?}", shape);
        }
        let per = arr.len();
        let src = arr.into_raw_vec_and_offset().0;
        let mut out = Vec::with_capacity(batch * per);
        for _ in 0..batch {
            out.extend_from_slice(&src);
        }
        let mut new_shape = shape;
        new_shape[0] = batch;
        Ok(Value::from_array(ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&new_shape),
            out,
        )?)?
        .into_dyn())
    }

    fn encode_images(&mut self, engines: &mut Engines, xs: &[Image]) -> Result<[DynValue; 4]> {
        let xs_ = elapsed_module!(
            "Sam3Image",
            "encode-images/preprocess",
            self.image_processor.process(xs)?
        );
        let ys = elapsed_module!(
            "Sam3Image",
            "encode-images/vision-encoder",
            engines.run(&Module::VisualEncoder, &xs_)?
        );

        // Keep the batched tensors as-is (batch dimension == xs.len()).
        // This is critical for efficiently decoding few-images + many-prompts.
        elapsed_module!("Sam3Image", "encode-images/fpn", {
            let out = ys.into_inner();
            let fpn: Vec<DynValue> = (0..4)
                .map(|i| {
                    out[i]
                        .view()
                        .try_upgrade()
                        .map_err(|_| anyhow!("Upgrade failed"))
                })
                .collect::<Result<_>>()?;
            fpn.try_into().map_err(|_| anyhow!("Expected 4 features"))
        })
    }

    fn encode_texts(
        &mut self,
        engines: &mut Engines,
        texts: &[&str],
    ) -> Result<Vec<(Arc<DynValue>, Arc<DynValue>)>> {
        let mut res = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.text_batch) {
            use ort::memory::AllocationDevice;
            use ort::tensor::TensorElementType as TE;
            use ort::value::ValueType;

            let encs = self.text_processor.encode_texts(chunk, true)?;
            let (n, l) = (chunk.len(), encs[0].get_ids().len());
            let ids = X::from_shape_vec(
                &[n, l],
                encs.iter()
                    .flat_map(|e| e.get_ids().iter().map(|&i| i as f32))
                    .collect(),
            )?;
            let mask = X::from_shape_vec(
                &[n, l],
                encs.iter()
                    .flat_map(|e| e.get_attention_mask().iter().map(|&m| m as f32))
                    .collect(),
            )?;
            let ys = elapsed_module!(
                "Sam3Image",
                "textual-encoder",
                engines.run(&Module::TextualEncoder, inputs![ids, mask]?)?
            );

            let out = ys.into_inner();
            // batch=1: 直接复用engine输出
            if n == 1 {
                let f = out[0]
                    .view()
                    .try_upgrade()
                    .map_err(|_| anyhow!("Upgrade failed"))?;
                let m = out[1]
                    .view()
                    .try_upgrade()
                    .map_err(|_| anyhow!("Upgrade failed"))?;
                res.push((Arc::new(f), Arc::new(m)));
            } else {
                // batch>1: 需要slice
                let (f_full, m_full) = {
                    let f = out[0]
                        .view()
                        .try_upgrade()
                        .map_err(|_| anyhow!("Upgrade failed"))?;
                    let f = {
                        let tensor = f
                            .downcast::<ort::value::DynTensorValueType>()
                            .map_err(|_| anyhow!("Failed to downcast to tensor"))?;
                        let tensor = if tensor.memory_info().is_cpu_accessible() {
                            tensor
                        } else {
                            tensor
                                .to(AllocationDevice::CPU, 0)
                                .or_else(|_| tensor.to(AllocationDevice::CUDA_PINNED, 0))?
                        };
                        tensor.into_dyn()
                    };
                    let m = out[1]
                        .view()
                        .try_upgrade()
                        .map_err(|_| anyhow!("Upgrade failed"))?;
                    let m = {
                        let tensor = m
                            .downcast::<ort::value::DynTensorValueType>()
                            .map_err(|_| anyhow!("Failed to downcast to tensor"))?;
                        let tensor = if tensor.memory_info().is_cpu_accessible() {
                            tensor
                        } else {
                            tensor
                                .to(AllocationDevice::CPU, 0)
                                .or_else(|_| tensor.to(AllocationDevice::CUDA_PINNED, 0))?
                        };
                        tensor.into_dyn()
                    };
                    (f, m)
                };
                let m_arr = Self::extract_f32(&m_full)?;
                let m_bool = m_arr.mapv(|x| x != 0.0);

                match f_full.dtype() {
                    ValueType::Tensor { ty, .. } => match ty {
                        TE::Float16 => {
                            let f_arr = f_full.try_extract_array::<f16>()?.into_dyn();
                            for i in 0..n {
                                res.push((
                                    Arc::new(
                                        Value::from_array(
                                            f_arr.slice_axis(Axis(0), (i..i + 1).into()).to_owned(),
                                        )?
                                        .into_dyn(),
                                    ),
                                    Arc::new(
                                        Value::from_array(
                                            m_bool
                                                .slice_axis(Axis(0), (i..i + 1).into())
                                                .to_owned(),
                                        )?
                                        .into_dyn(),
                                    ),
                                ));
                            }
                        }
                        TE::Float32 => {
                            let f_arr = f_full.try_extract_array::<f32>()?.into_dyn();
                            for i in 0..n {
                                res.push((
                                    Arc::new(
                                        Value::from_array(
                                            f_arr.slice_axis(Axis(0), (i..i + 1).into()).to_owned(),
                                        )?
                                        .into_dyn(),
                                    ),
                                    Arc::new(
                                        Value::from_array(
                                            m_bool
                                                .slice_axis(Axis(0), (i..i + 1).into())
                                                .to_owned(),
                                        )?
                                        .into_dyn(),
                                    ),
                                ));
                            }
                        }
                        _ => bail!("Unsupported text_features tensor type: {:?}", ty),
                    },
                    _ => bail!("Unsupported text_features value type"),
                }
            }
        }
        Ok(res)
    }

    fn forward_image(
        &mut self,
        engines: &mut Engines,
        xs: &[Image],
        prompts: &[Sam3Prompt],
    ) -> Result<Vec<Y>> {
        if xs.is_empty() || prompts.is_empty() {
            return Ok(vec![]);
        }

        // encode images
        let fpn = elapsed_module!(
            "Sam3Image",
            "encode-images",
            self.encode_images(engines, xs)?
        );

        // encode all text prompts
        let texts: Vec<_> = elapsed_module!("Sam3Image", "encode-text-prompts", {
            // update class names
            self.names = prompts.iter().map(|p| p.text.clone()).collect();

            let mut uncached = Vec::new();
            let mut seen = HashSet::new();
            for p in prompts {
                if seen.insert(&p.text) && !self.text_cache.contains(&p.text) {
                    uncached.push(p.text.as_str());
                }
            }
            if !uncached.is_empty() {
                for (text, feat) in uncached.iter().zip(self.encode_texts(engines, &uncached)?) {
                    self.text_cache.put(text.to_string(), feat);
                }
            }
            prompts
                .iter()
                .map(|p| {
                    let (f, m) = self.text_cache.get(&p.text).unwrap();
                    (Arc::clone(f), Arc::clone(m))
                })
                .collect()
        });

        elapsed_module!("Sam3Image", "decode-all-images", {
            let infos = self.image_processor.images_transform_info().to_vec();

            let (mh, mw) = (
                self.image_processor.image_height() as f32,
                self.image_processor.image_width() as f32,
            );

            let batch = xs.len();
            let mut results = vec![Y::default(); batch];

            // Fast path: batch==1, no need to replicate text or handle batched postprocess.
            if batch == 1 {
                let info = &infos[0];
                let (sx, sy) = (mw / info.width_src as f32, mh / info.height_src as f32);
                elapsed_module!("Sam3Image", "decode-all-prompts", {
                    for (pi, prompt) in prompts.iter().enumerate() {
                        let (text_feat, text_mask) = &texts[pi];
                        let (boxes, labels) = if prompt.should_use_geometry() {
                            let b = prompt.normalized_boxes_scaled(sx, sy, mw, mh);
                            (
                                X::from_shape_vec(
                                    &[1, b.len(), 4],
                                    b.iter().flat_map(|b| b.iter().copied()).collect(),
                                )?,
                                X::<i64>::from_shape_vec_generic(
                                    &[1, b.len()],
                                    prompt.box_labels(),
                                )?,
                            )
                        } else {
                            (
                                X::from_shape_vec(&[1, 1, 4], vec![0.; 4])?,
                                X::<i64>::from_shape_vec_generic(&[1, 1], vec![-10_i64])?,
                            )
                        };

                        let ys = elapsed_module!(
                            "Sam3Image",
                            "decoder",
                            engines.run(
                                &Module::Decoder,
                                inputs![
                                    &fpn[0],
                                    &fpn[1],
                                    &fpn[2],
                                    &fpn[3],
                                    text_feat.as_ref(),
                                    text_mask.as_ref(),
                                    boxes.view(),
                                    labels.view()
                                ]?
                            )?
                        );

                        let r = elapsed_module!(
                            "Sam3Image",
                            "postprocess",
                            self.postprocess(
                                &ys,
                                info.height_src as _,
                                info.width_src as _,
                                0,
                                pi,
                                prompt.class_name(),
                            )?
                        );
                        results[0].masks.extend(r.masks);
                        results[0].hbbs.extend(r.hbbs);
                    }
                });
                return Ok(results);
            }

            // Few-images + many-prompts path:
            // Run decoder once per prompt with batch==#images.
            elapsed_module!("Sam3Image", "decode-all-prompts", {
                for (pi, prompt) in prompts.iter().enumerate() {
                    let (text_feat, text_mask) = &texts[pi];

                    let feat_batched = Self::repeat_text_feat_to_batch(text_feat.as_ref(), batch)?;
                    let mask_batched = Self::repeat_text_mask_to_batch(text_mask.as_ref(), batch)?;

                    let (boxes, labels) = if prompt.should_use_geometry() {
                        let labels0 = prompt.box_labels();
                        let nb = labels0.len();
                        let mut boxes_flat = Vec::with_capacity(batch * nb * 4);
                        let mut labels_flat = Vec::with_capacity(batch * nb);
                        for info in infos.iter() {
                            let (sx, sy) =
                                (mw / info.width_src as f32, mh / info.height_src as f32);
                            let b = prompt.normalized_boxes_scaled(sx, sy, mw, mh);
                            boxes_flat.extend(b.iter().flat_map(|bb| bb.iter().copied()));
                            labels_flat.extend(labels0.iter().copied());
                        }
                        (
                            X::from_shape_vec(&[batch, nb, 4], boxes_flat)?,
                            X::<i64>::from_shape_vec_generic(&[batch, nb], labels_flat)?,
                        )
                    } else {
                        (
                            X::from_shape_vec(&[batch, 1, 4], vec![0.0; batch * 4])?,
                            X::<i64>::from_shape_vec_generic(&[batch, 1], vec![-10_i64; batch])?,
                        )
                    };

                    let ys = elapsed_module!(
                        "Sam3Image",
                        "decoder",
                        engines.run(
                            &Module::Decoder,
                            inputs![
                                &fpn[0],
                                &fpn[1],
                                &fpn[2],
                                &fpn[3],
                                &feat_batched,
                                &mask_batched,
                                boxes.view(),
                                labels.view()
                            ]?
                        )?
                    );

                    for (img_idx, info) in infos.iter().enumerate() {
                        let r = elapsed_module!(
                            "Sam3Image",
                            "postprocess",
                            self.postprocess(
                                &ys,
                                info.height_src as _,
                                info.width_src as _,
                                img_idx,
                                pi,
                                prompt.class_name(),
                            )?
                        );
                        results[img_idx].masks.extend(r.masks);
                        results[img_idx].hbbs.extend(r.hbbs);
                    }
                }
            });

            Ok(results)
        })
    }

    fn postprocess(
        &self,
        outputs: &Xs,
        image_height: usize,
        image_width: usize,
        batch_index: usize,
        class_id: usize,
        class_name: &str,
    ) -> Result<Y> {
        let masks = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow!("Failed to get masks"))?;
        let boxes = outputs
            .get::<f32>(1)
            .ok_or_else(|| anyhow!("Failed to get boxes"))?;
        let logits = outputs
            .get::<f32>(2)
            .ok_or_else(|| anyhow!("Failed to get logits"))?;
        let presence = outputs
            .get::<f32>(3)
            .ok_or_else(|| anyhow!("Failed to get presence"))?;

        let presence_score = 1.0 / (1.0 + (-presence.0[[batch_index, 0]]).exp());
        let scores: Array1<f32> = logits
            .0
            .slice(s![batch_index, ..])
            .mapv(|x| 1.0 / (1.0 + (-x).exp()) * presence_score);
        let valid: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s >= self.conf[0])
            .map(|(i, _)| i)
            .collect();
        if valid.is_empty() {
            return Ok(Y::default());
        }

        let res: Vec<_> = valid
            .into_par_iter()
            .filter_map(|idx| {
                let mask_view = masks.0.slice(s![batch_index, idx, .., ..]);
                let (mh, mw) = mask_view.dim();

                let src = match mask_view.as_slice_memory_order() {
                    Some(s) => std::borrow::Cow::Borrowed(s),
                    None => {
                        std::borrow::Cow::Owned(mask_view.to_owned().into_raw_vec_and_offset().0)
                    }
                };

                let luma = Ops::interpolate_1d_u8(
                    src.as_ref(),
                    mw as _,
                    mh as _,
                    image_width as _,
                    image_height as _,
                    false,
                )
                .ok()?;

                let mask = Mask::new(&luma, image_width as u32, image_height as u32)
                    .ok()?
                    .with_id(class_id)
                    .with_name(class_name)
                    .with_confidence(scores[idx]);

                let hbb = Hbb::default()
                    .with_xyxy(
                        boxes.0[[batch_index, idx, 0]] * image_width as f32,
                        boxes.0[[batch_index, idx, 1]] * image_height as f32,
                        boxes.0[[batch_index, idx, 2]] * image_width as f32,
                        boxes.0[[batch_index, idx, 3]] * image_height as f32,
                    )
                    .with_confidence(scores[idx])
                    .with_id(class_id)
                    .with_name(class_name);

                Some((mask, hbb))
            })
            .collect();

        let (y_masks, y_hbbs): (Vec<_>, Vec<_>) = res.into_iter().unzip();
        Ok(Y::default().with_masks(&y_masks).with_hbbs(&y_hbbs))
    }

    pub fn with_text_cache_size(mut self, max_size: usize) -> Self {
        self.text_cache
            .resize(NonZeroUsize::new(max_size.max(1)).unwrap());
        self
    }

    pub fn clear_text_cache(&mut self) {
        self.text_cache.clear();
    }
}

impl Model for Sam3Image {
    type Input<'a> = (&'a [Image], &'a [Sam3Prompt]);
    fn batch(&self) -> usize {
        self.vision_batch
    }
    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let decoder = Engine::from_config(config.take_module(&Module::Decoder)?)?;
        let visual_encoder = Engine::from_config(config.take_module(&Module::VisualEncoder)?)?;
        let textual_encoder = Engine::from_config(config.take_module(&Module::TextualEncoder)?)?;

        let (vision_batch, text_batch, decoder_batch) = (
            visual_encoder.batch().opt(),
            textual_encoder.batch().opt(),
            decoder.batch().opt(),
        );
        let (height, width) = (
            visual_encoder.try_height().unwrap_or(&1008.into()).opt(),
            visual_encoder.try_width().unwrap_or(&1008.into()).opt(),
        );
        let conf = DynConf::new_or_default(config.class_confs(), 1);

        let model = Self {
            vision_batch,
            text_batch,
            decoder_batch,
            image_processor: ImageProcessor::from_config(config.image_processor)?
                .with_image_width(width as _)
                .with_image_height(height as _),
            text_processor: TextProcessor::from_config(config.text_processor)?,
            conf,
            text_cache: LruCache::new(NonZeroUsize::new(16).unwrap()),
            names: config.inference.class_names,
            spec: "sam3-image".to_string(),
        };

        let mut engines = Engines::new();
        engines.insert(Module::VisualEncoder, visual_encoder);
        engines.insert(Module::TextualEncoder, textual_encoder);
        engines.insert(Module::Decoder, decoder);

        Ok((model, engines))
    }

    fn run(&mut self, engines: &mut Engines, (images, prompts): Self::Input<'_>) -> Result<Vec<Y>> {
        self.forward_image(engines, images, prompts)
    }
}
