//! Options for everthing

use aksr::Builder;
use anyhow::Result;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::{
    models::{SamKind, YOLOPredsFormat},
    DType, Device, Engine, Hub, Iiix, Kind, LogitsSampler, MinOptMax, Processor, ResizeMode, Scale,
    Task, Version,
};

/// Options for building models and inference
#[derive(Builder, Debug, Clone)]
pub struct Options {
    // Model configs
    pub model_file: String,
    pub model_name: &'static str,
    pub model_device: Device,
    pub model_dtype: DType,
    pub model_version: Option<Version>,
    pub model_task: Option<Task>,
    pub model_scale: Option<Scale>,
    pub model_kind: Option<Kind>,
    pub model_iiixs: Vec<Iiix>,
    pub model_spec: String,
    pub model_num_dry_run: usize,
    pub trt_fp16: bool,
    pub profile: bool,

    // Processor configs
    #[args(setter = false)]
    pub image_width: u32,
    #[args(setter = false)]
    pub image_height: u32,
    pub resize_mode: ResizeMode,
    pub resize_filter: &'static str,
    pub padding_value: u8,
    pub letterbox_center: bool,
    pub normalize: bool,
    pub image_std: Vec<f32>,
    pub image_mean: Vec<f32>,
    pub nchw: bool,
    pub unsigned: bool,

    // Names
    pub class_names: Option<Vec<String>>,
    pub class_names_2: Option<Vec<String>>,
    pub class_names_3: Option<Vec<String>>,
    pub keypoint_names: Option<Vec<String>>,
    pub keypoint_names_2: Option<Vec<String>>,
    pub keypoint_names_3: Option<Vec<String>>,
    pub text_names: Option<Vec<String>>,
    pub text_names_2: Option<Vec<String>>,
    pub text_names_3: Option<Vec<String>>,
    pub category_names: Option<Vec<String>>,
    pub category_names_2: Option<Vec<String>>,
    pub category_names_3: Option<Vec<String>>,

    // Confs
    pub class_confs: Vec<f32>,
    pub class_confs_2: Vec<f32>,
    pub class_confs_3: Vec<f32>,
    pub keypoint_confs: Vec<f32>,
    pub keypoint_confs_2: Vec<f32>,
    pub keypoint_confs_3: Vec<f32>,
    pub text_confs: Vec<f32>,
    pub text_confs_2: Vec<f32>,
    pub text_confs_3: Vec<f32>,

    // For classification
    pub apply_softmax: Option<bool>,

    // For detection
    #[args(alias = "nc")]
    pub num_classes: Option<usize>,
    #[args(alias = "nk")]
    pub num_keypoints: Option<usize>,
    #[args(alias = "nm")]
    pub num_masks: Option<usize>,
    pub iou: Option<f32>,
    pub iou_2: Option<f32>,
    pub iou_3: Option<f32>,
    pub apply_nms: Option<bool>,
    pub find_contours: bool,
    pub yolo_preds_format: Option<YOLOPredsFormat>,
    pub classes_excluded: Vec<usize>,
    pub classes_retained: Vec<usize>,
    pub min_width: Option<f32>,
    pub min_height: Option<f32>,

    // Language models related
    pub model_max_length: Option<u64>,
    pub tokenizer_file: Option<String>,
    pub config_file: Option<String>,
    pub special_tokens_map_file: Option<String>,
    pub tokenizer_config_file: Option<String>,
    pub generation_config_file: Option<String>,
    pub vocab_file: Option<String>, // vocab.json file
    pub vocab_txt: Option<String>,  // vacab.txt file, not kv pairs
    pub temperature: f32,
    pub topp: f32,

    // For DB
    pub unclip_ratio: Option<f32>,
    pub binary_thresh: Option<f32>,

    // For SAM
    pub sam_kind: Option<SamKind>,
    pub low_res_mask: Option<bool>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model_file: Default::default(),
            model_name: Default::default(),
            model_version: Default::default(),
            model_task: Default::default(),
            model_scale: Default::default(),
            model_kind: Default::default(),
            model_device: Device::Cpu(0),
            model_dtype: DType::Auto,
            model_spec: Default::default(),
            model_iiixs: Default::default(),
            model_num_dry_run: 3,
            trt_fp16: true,
            profile: false,
            normalize: true,
            image_mean: vec![],
            image_std: vec![],
            image_height: 640,
            image_width: 640,
            padding_value: 114,
            resize_mode: ResizeMode::FitExact,
            resize_filter: "Bilinear",
            letterbox_center: false,
            nchw: true,
            unsigned: false,
            class_names: None,
            class_names_2: None,
            class_names_3: None,
            category_names: None,
            category_names_2: None,
            category_names_3: None,
            keypoint_names: None,
            keypoint_names_2: None,
            keypoint_names_3: None,
            text_names: None,
            text_names_2: None,
            text_names_3: None,
            class_confs: vec![0.3f32],
            class_confs_2: vec![0.3f32],
            class_confs_3: vec![0.3f32],
            keypoint_confs: vec![0.3f32],
            keypoint_confs_2: vec![0.5f32],
            keypoint_confs_3: vec![0.5f32],
            text_confs: vec![0.4f32],
            text_confs_2: vec![0.4f32],
            text_confs_3: vec![0.4f32],
            apply_softmax: Some(false),
            num_classes: None,
            num_keypoints: None,
            num_masks: None,
            iou: None,
            iou_2: None,
            iou_3: None,
            find_contours: false,
            yolo_preds_format: None,
            classes_excluded: vec![],
            classes_retained: vec![],
            apply_nms: None,
            model_max_length: None,
            tokenizer_file: None,
            config_file: None,
            special_tokens_map_file: None,
            tokenizer_config_file: None,
            generation_config_file: None,
            vocab_file: None,
            vocab_txt: None,
            min_width: None,
            min_height: None,
            unclip_ratio: Some(1.5),
            binary_thresh: Some(0.2),
            sam_kind: None,
            low_res_mask: None,
            temperature: 1.,
            topp: 0.,
        }
    }
}

impl Options {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn to_engine(&self) -> Result<Engine> {
        Engine {
            file: self.model_file.clone(),
            spec: self.model_spec.clone(),
            device: self.model_device,
            trt_fp16: self.trt_fp16,
            iiixs: self.model_iiixs.clone(),
            num_dry_run: self.model_num_dry_run,
            ..Default::default()
        }
        .build()
    }

    pub fn to_processor(&self) -> Result<Processor> {
        let logits_sampler = LogitsSampler::new()
            .with_temperature(self.temperature)
            .with_topp(self.topp);

        // try to build tokenizer
        let tokenizer = match self.model_kind {
            Some(Kind::Language) | Some(Kind::VisionLanguage) => Some(self.try_build_tokenizer()?),
            _ => None,
        };

        // try to build vocab from `vocab.txt`
        let vocab: Vec<String> = match &self.vocab_txt {
            Some(x) => {
                let file = if !std::path::PathBuf::from(&x).exists() {
                    Hub::default().try_fetch(&format!("{}/{}", self.model_name, x))?
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
            image_width: self.image_width,
            image_height: self.image_height,
            resize_mode: self.resize_mode.clone(),
            resize_filter: self.resize_filter,
            padding_value: self.padding_value,
            do_normalize: self.normalize,
            image_mean: self.image_mean.clone(),
            image_std: self.image_std.clone(),
            nchw: self.nchw,
            unsigned: self.unsigned,
            tokenizer,
            vocab,
            logits_sampler: Some(logits_sampler),
            ..Default::default()
        })
    }

    pub fn commit(mut self) -> Result<Self> {
        // Identify the local model or fetch the remote model

        if std::path::PathBuf::from(&self.model_file).exists() {
            // Local
            self.model_spec = format!(
                "{}/{}",
                self.model_name,
                crate::try_fetch_stem(&self.model_file)?
            );
        } else {
            // Remote
            if self.model_file.is_empty() && self.model_name.is_empty() {
                anyhow::bail!("Neither `model_name` nor `model_file` were specified. Faild to fetch model from remote.")
            }

            // Load
            match Hub::is_valid_github_release_url(&self.model_file) {
                Some((owner, repo, tag, _file_name)) => {
                    let stem = crate::try_fetch_stem(&self.model_file)?;
                    self.model_spec =
                        format!("{}/{}-{}-{}-{}", self.model_name, owner, repo, tag, stem);
                    self.model_file = Hub::default().try_fetch(&self.model_file)?;
                }
                None => {
                    // special yolo case
                    if self.model_file.is_empty() && self.model_name == "yolo" {
                        // [version]-[scale]-[task]
                        let mut y = String::new();
                        if let Some(x) = self.model_version() {
                            y.push_str(&x.to_string());
                        }
                        if let Some(x) = self.model_scale() {
                            y.push_str(&format!("-{}", x));
                        }
                        if let Some(x) = self.model_task() {
                            y.push_str(&format!("-{}", x.yolo_str()));
                        }
                        y.push_str(".onnx");
                        self.model_file = y;
                    }

                    // append dtype to model file
                    match self.model_dtype {
                        d @ (DType::Auto | DType::Fp32) => {
                            if self.model_file.is_empty() {
                                self.model_file = format!("{}.onnx", d);
                            }
                        }
                        dtype => {
                            if self.model_file.is_empty() {
                                self.model_file = format!("{}.onnx", dtype);
                            } else {
                                let pos = self.model_file.len() - 5; // .onnx
                                let suffix = self.model_file.split_off(pos);
                                self.model_file =
                                    format!("{}-{}{}", self.model_file, dtype, suffix);
                            }
                        }
                    }

                    let stem = crate::try_fetch_stem(&self.model_file)?;
                    self.model_spec = format!("{}/{}", self.model_name, stem);
                    self.model_file = Hub::default()
                        .try_fetch(&format!("{}/{}", self.model_name, self.model_file))?;
                }
            }

            // let stem = crate::try_fetch_stem(&self.model_file)?;
            // self.model_spec = format!("{}/{}", self.model_name, stem);
            // self.model_file =
            //     Hub::default().try_fetch(&format!("{}/{}", self.model_name, self.model_file))?;
        }

        Ok(self)
    }

    pub fn with_batch_size(mut self, x: usize) -> Self {
        self.model_iiixs.push(Iiix::from((0, 0, x.into())));
        self
    }

    pub fn with_image_height(mut self, x: u32) -> Self {
        self.image_height = x;
        self.model_iiixs.push(Iiix::from((0, 2, x.into())));
        self
    }

    pub fn with_image_width(mut self, x: u32) -> Self {
        self.image_width = x;
        self.model_iiixs.push(Iiix::from((0, 3, x.into())));
        self
    }

    pub fn with_model_ixx(mut self, i: usize, ii: usize, x: MinOptMax) -> Self {
        self.model_iiixs.push(Iiix::from((i, ii, x)));
        self
    }

    pub fn exclude_classes(mut self, xs: &[usize]) -> Self {
        self.classes_retained.clear();
        self.classes_excluded.extend_from_slice(xs);
        self
    }

    pub fn retain_classes(mut self, xs: &[usize]) -> Self {
        self.classes_excluded.clear();
        self.classes_retained.extend_from_slice(xs);
        self
    }

    pub fn try_build_tokenizer(&self) -> Result<Tokenizer> {
        let mut hub = Hub::default();
        // config file
        // TODO: save configs?
        let pad_id = match hub.try_fetch(
            self.tokenizer_config_file
                .as_ref()
                .unwrap_or(&format!("{}/config.json", self.model_name)),
        ) {
            Ok(x) => {
                let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(x)?)?;
                config["pad_token_id"].as_u64().unwrap_or(0) as u32
            }
            Err(_err) => 0u32,
        };

        // tokenizer_config file
        let mut max_length = None;
        let mut pad_token = String::from("[PAD]");
        match hub.try_fetch(
            self.tokenizer_config_file
                .as_ref()
                .unwrap_or(&format!("{}/tokenizer_config.json", self.model_name)),
        ) {
            Err(_) => {}
            Ok(x) => {
                let tokenizer_config: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(x)?)?;
                max_length = tokenizer_config["model_max_length"].as_u64();
                pad_token = tokenizer_config["pad_token"]
                    .as_str()
                    .unwrap_or("[PAD]")
                    .to_string();
            }
        }

        // tokenizer file
        let mut tokenizer: tokenizers::Tokenizer = tokenizers::Tokenizer::from_file(
            hub.try_fetch(
                self.tokenizer_file
                    .as_ref()
                    .unwrap_or(&format!("{}/tokenizer.json", self.model_name)),
            )?,
        )
        .map_err(|_| anyhow::anyhow!("No `tokenizer.json` found"))?;

        // TODO: padding
        // if `max_length` specified: use `Fixed` strategy
        // else: use `BatchLongest` strategy
        // TODO: if sequence_length is dynamic, `BatchLongest` is fine
        let tokenizer = match self.model_max_length {
            Some(n) => {
                let n = match max_length {
                    None => n,
                    Some(x) => x.min(n),
                };
                tokenizer
                    .with_padding(Some(PaddingParams {
                        strategy: PaddingStrategy::Fixed(n as _),
                        pad_token,
                        pad_id,
                        ..Default::default()
                    }))
                    .clone()
            }
            None => match max_length {
                Some(n) => tokenizer
                    .with_padding(Some(PaddingParams {
                        strategy: PaddingStrategy::BatchLongest,
                        pad_token,
                        pad_id,
                        ..Default::default()
                    }))
                    .with_truncation(Some(TruncationParams {
                        max_length: n as _,
                        ..Default::default()
                    }))
                    .map_err(|err| anyhow::anyhow!("Failed to truncate: {}", err))?
                    .clone(),
                None => tokenizer
                    .with_padding(Some(PaddingParams {
                        strategy: PaddingStrategy::BatchLongest,
                        pad_token,
                        pad_id,
                        ..Default::default()
                    }))
                    .clone(),
            },
        };

        // TODO: generation_config.json & special_tokens_map file

        Ok(tokenizer.into())
    }
}
