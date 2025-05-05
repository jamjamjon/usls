use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Array};
use rayon::prelude::*;
use std::sync::Mutex;
use tokenizers::{Encoding, Tokenizer};

use crate::{Hub, Image, ImageTransformInfo, LogitsSampler, ProcessorConfig, ResizeMode, X};

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
    pub logits_sampler: Option<LogitsSampler>, // TODO: not just for language models, but for vision models. Provide Topk, TopP, BeamSearch, ArgMax, ArgMix, etc.
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
        }
    }
}

impl Processor {
    pub fn try_from_config(config: ProcessorConfig) -> Result<Self> {
        let logits_sampler = LogitsSampler::new()
            .with_temperature(config.temperature)
            .with_topp(config.topp);

        // try to build tokenizer
        // let tokenizer = match self.model_kind {
        //     Some(Kind::Language) | Some(Kind::VisionLanguage) => Some(self.try_build_tokenizer()?),
        //     _ => None,
        // };
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
            padding_value: config.padding_value,
            do_normalize: config.normalize,
            image_mean: config.image_mean.clone(),
            image_std: config.image_std.clone(),
            nchw: config.nchw,
            unsigned: config.unsigned,
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
        // self.reset_image0_status();
        let (mut x, images_transform_info) = self.par_resize(xs)?;
        self.images_transform_info = images_transform_info;

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
        self.tokenizer
            .as_ref()
            .expect("No tokenizer specified in `Processor`")
            .encode(x, skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("Tokenizer encode error: {}", err))
    }

    pub fn encode_texts(&self, xs: &[&str], skip_special_tokens: bool) -> Result<Vec<Encoding>> {
        self.tokenizer
            .as_ref()
            .expect("No tokenizer specified in `Processor`")
            .encode_batch(xs.to_vec(), skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("Tokenizer encode_batch error: {}", err))
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
        self.tokenizer
            .as_ref()
            .expect("No tokenizer specified in `Processor`")
            .decode(ids, skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("Tokenizer decode error: {}", err))
    }

    pub fn decode_tokens_batch2(
        &self,
        ids: &[&[u32]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        self.tokenizer
            .as_ref()
            .expect("No tokenizer specified in `Processor`")
            .decode_batch(ids, skip_special_tokens)
            .map_err(|err| anyhow::anyhow!("Tokenizer decode_batch error: {}", err))
    }

    pub fn decode_tokens_batch(
        &self,
        ids: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        self.tokenizer
            .as_ref()
            .expect("No tokenizer specified in `Processor`")
            .decode_batch(
                &ids.iter().map(|x| x.as_slice()).collect::<Vec<_>>(),
                skip_special_tokens,
            )
            .map_err(|err| anyhow::anyhow!("Tokenizer decode_batch error: {}", err))
    }
}
