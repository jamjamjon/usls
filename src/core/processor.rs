use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Array};
use rayon::prelude::*;
use std::sync::Mutex;
use tokenizers::{Encoding, Tokenizer};

use crate::{Hub, Image, ImageTransformInfo, LogitsSampler, ProcessorConfig, ResizeMode, X};

/// Image and text processing pipeline with tokenization and transformation capabilities.
#[derive(Builder, Debug, Clone)]
pub struct Processor {
    pub image_width: u32,
    pub image_height: u32,
    pub images_transform_info: Vec<ImageTransformInfo>,
    pub resize_mode: ResizeMode,
    pub resize_filter: &'static str,
    pub padding_value: u8,
    pub do_normalize: bool,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub nchw: bool,
    pub tokenizer: Option<Tokenizer>,
    pub vocab: Vec<String>,
    pub unsigned: bool,
    pub logits_sampler: Option<LogitsSampler>,
    pub pad_image: bool,
    pub pad_size: usize,
    pub up_scale: f32,
    pub do_resize: bool,
}

impl Default for Processor {
    fn default() -> Self {
        Self {
            images_transform_info: vec![],
            image_width: 0,
            image_height: 0,
            resize_mode: ResizeMode::FitAdaptive,
            resize_filter: "Bilinear",
            padding_value: 114,
            do_normalize: true,
            image_mean: vec![],
            image_std: vec![],
            nchw: true,
            tokenizer: Default::default(),
            vocab: vec![],
            unsigned: false,
            logits_sampler: None,
            pad_image: false,
            pad_size: 8,
            up_scale: 2.,
            do_resize: true,
        }
    }
}

impl Processor {
    pub fn try_from_config(config: &ProcessorConfig) -> Result<Self> {
        let logits_sampler = LogitsSampler::new()
            .with_temperature(config.temperature)
            .with_topp(config.topp);

        // try to build tokenizer
        let tokenizer = config.try_build_tokenizer()?;

        // try to build vocab from `vocab.txt`
        let vocab: Vec<String> = match &config.vocab_txt {
            Some(x) => {
                let file = if !std::path::PathBuf::from(&x).exists() {
                    Hub::default().try_fetch(x)?
                } else {
                    x.to_string()
                };
                std::fs::read_to_string(file)?
                    .lines()
                    .map(|line| line.to_string())
                    .collect()
            }
            None => vec![],
        };

        Ok(Processor {
            image_width: config.image_width.unwrap_or_default(),
            image_height: config.image_height.unwrap_or_default(),
            resize_mode: config.resize_mode.clone(),
            resize_filter: config.resize_filter.unwrap_or("Bilinear"),
            do_resize: config.do_resize,
            padding_value: config.padding_value,
            do_normalize: config.normalize,
            image_mean: config.image_mean.clone(),
            image_std: config.image_std.clone(),
            nchw: config.nchw,
            unsigned: config.unsigned,
            pad_image: config.pad_image,
            pad_size: config.pad_size,
            up_scale: config.up_scale,
            tokenizer,
            vocab,
            logits_sampler: Some(logits_sampler),
            ..Default::default()
        })
    }

    pub fn reset_image0_status(&mut self) {
        self.images_transform_info.clear();
    }

    pub fn process_images(&mut self, xs: &[Image]) -> Result<X> {
        let mut x = if self.pad_image {
            if xs.len() != 1 {
                anyhow::bail!("When pad_image is true, only one image is allowed.");
            }
            let (image, images_transform_info) = xs[0].pad(self.pad_size)?;
            self.images_transform_info = vec![images_transform_info];
            Image::from(image).to_ndarray()?.insert_axis(0)?
        } else if self.do_resize {
            let (x, images_transform_info) = self.par_resize(xs)?;
            self.images_transform_info = images_transform_info;
            x
        } else {
            anyhow::bail!(
                "When pad_image and do_resize are both false, at least one image is required."
            );
        };

        if self.do_normalize {
            x = x.normalize(0., 255.)?;
        }
        if !self.image_std.is_empty() && !self.image_mean.is_empty() {
            x = x.standardize(&self.image_mean, &self.image_std, 3)?;
        }
        if self.nchw {
            x = x.nhwc2nchw()?;
        }
        if self.unsigned {
            x = x.unsigned();
        }

        Ok(x)
    }

    pub fn par_resize(&self, xs: &[Image]) -> Result<(X, Vec<ImageTransformInfo>)> {
        match xs.len() {
            0 => anyhow::bail!("Found no input images."),
            1 => {
                let (image, trans_info) = xs[0].resize_with_info(
                    self.image_width,
                    self.image_height,
                    self.resize_filter,
                    &self.resize_mode,
                    self.padding_value,
                )?;

                let y = image.to_ndarray()?.insert_axis(0)?;
                Ok((y, vec![trans_info]))
            }
            _ => {
                let ys = Mutex::new(
                    Array::zeros((
                        xs.len(),
                        self.image_height as usize,
                        self.image_width as usize,
                        3,
                    ))
                    .into_dyn(),
                );

                let results: Result<Vec<ImageTransformInfo>> = xs
                    .par_iter()
                    .enumerate()
                    .map(|(idx, x)| {
                        let (image, trans_info) = x.resize_with_info(
                            self.image_width,
                            self.image_height,
                            self.resize_filter,
                            &self.resize_mode,
                            self.padding_value,
                        )?;

                        let y = image.to_ndarray()?;
                        {
                            let mut ys_guard = ys
                                .lock()
                                .map_err(|e| anyhow::anyhow!("Mutex lock error: {e}"))?;
                            ys_guard.slice_mut(s![idx, .., .., ..]).assign(&y);
                        }

                        Ok(trans_info)
                    })
                    .collect();

                let ys_inner = ys
                    .into_inner()
                    .map_err(|e| anyhow::anyhow!("Mutex into_inner error: {e}"))?;

                Ok((ys_inner.into(), results?))
            }
        }
    }

    pub fn encode_text(&self, x: &str, skip_special_tokens: bool) -> Result<Encoding> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "No tokenizer configured in Processor. Please initialize with a tokenizer."
            )
        })?;

        tokenizer.encode(x, skip_special_tokens).map_err(|err| {
            anyhow::anyhow!(
                "Failed to encode text '{}': {}",
                x.chars().take(50).collect::<String>(),
                err
            )
        })
    }

    pub fn encode_texts(&self, xs: &[&str], skip_special_tokens: bool) -> Result<Vec<Encoding>> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "No tokenizer configured in Processor. Please initialize with a tokenizer."
            )
        })?;

        tokenizer
            .encode_batch(xs.to_vec(), skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("Failed to encode batch of {} texts: {}", xs.len(), err))
    }

    pub fn encode_text_ids(&self, x: &str, skip_special_tokens: bool) -> Result<Vec<f32>> {
        let ids: Vec<f32> = if x.is_empty() {
            vec![0.0f32]
        } else {
            self.encode_text(x, skip_special_tokens)?
                .get_ids()
                .iter()
                .map(|x| *x as f32)
                .collect()
        };

        Ok(ids)
    }

    pub fn encode_texts_ids(
        &self,
        xs: &[&str],
        skip_special_tokens: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let ids: Vec<Vec<f32>> = if xs.is_empty() {
            vec![vec![0.0f32]]
        } else {
            self.encode_texts(xs, skip_special_tokens)?
                .into_iter()
                .map(|encoding| encoding.get_ids().iter().map(|x| *x as f32).collect())
                .collect()
        };

        Ok(ids)
    }

    pub fn encode_text_tokens(&self, x: &str, skip_special_tokens: bool) -> Result<Vec<String>> {
        Ok(self
            .encode_text(x, skip_special_tokens)?
            .get_tokens()
            .to_vec())
    }

    pub fn encode_texts_tokens(
        &self,
        xs: &[&str],
        skip_special_tokens: bool,
    ) -> Result<Vec<Vec<String>>> {
        Ok(self
            .encode_texts(xs, skip_special_tokens)?
            .into_iter()
            .map(|encoding| encoding.get_tokens().to_vec())
            .collect())
    }

    pub fn decode_tokens(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "No tokenizer configured in Processor. Please initialize with a tokenizer."
            )
        })?;

        tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("Failed to decode {} token IDs: {}", ids.len(), err))
    }

    pub fn decode_tokens_batch2(
        &self,
        ids: &[&[u32]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "No tokenizer configured in Processor. Please initialize with a tokenizer."
            )
        })?;

        tokenizer
            .decode_batch(ids, skip_special_tokens)
            .map_err(|err| {
                anyhow::anyhow!(
                    "Failed to decode batch of {} token sequences: {}",
                    ids.len(),
                    err
                )
            })
    }

    pub fn decode_tokens_batch(
        &self,
        ids: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "No tokenizer configured in Processor. Please initialize with a tokenizer."
            )
        })?;

        tokenizer
            .decode_batch(
                &ids.iter().map(|x| x.as_slice()).collect::<Vec<_>>(),
                skip_special_tokens,
            )
            .map_err(|err| {
                anyhow::anyhow!(
                    "Failed to decode batch of {} token vectors: {}",
                    ids.len(),
                    err
                )
            })
    }
}
