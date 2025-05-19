use aksr::Builder;

use crate::{
    impl_model_config_methods, impl_process_config_methods,
    models::{SamKind, YOLOPredsFormat},
    EngineConfig, ProcessorConfig, Scale, Task, Version,
};

/// ModelConfig for building models and inference
#[derive(Builder, Debug, Clone)]
pub struct ModelConfig {
    // Basics
    pub name: &'static str,
    pub version: Option<Version>,
    pub task: Option<Task>,
    pub scale: Option<Scale>,

    // Engines
    pub model: EngineConfig,
    pub visual: EngineConfig,
    pub textual: EngineConfig,
    pub encoder: EngineConfig,
    pub decoder: EngineConfig,
    pub visual_encoder: EngineConfig,
    pub textual_encoder: EngineConfig,
    pub visual_decoder: EngineConfig,
    pub textual_decoder: EngineConfig,
    pub textual_decoder_merged: EngineConfig,
    pub size_encoder: EngineConfig,
    pub size_decoder: EngineConfig,
    pub coord_encoder: EngineConfig,
    pub coord_decoder: EngineConfig,
    pub visual_projection: EngineConfig,
    pub textual_projection: EngineConfig,

    // Processor
    pub processor: ProcessorConfig,

    // Others
    pub class_names: Vec<String>,
    pub keypoint_names: Vec<String>,
    pub text_names: Vec<String>,
    pub class_confs: Vec<f32>,
    pub keypoint_confs: Vec<f32>,
    pub text_confs: Vec<f32>,
    pub apply_softmax: Option<bool>,
    pub topk: Option<usize>,
    #[args(aka = "nc")]
    pub num_classes: Option<usize>,
    #[args(aka = "nk")]
    pub num_keypoints: Option<usize>,
    #[args(aka = "nm")]
    pub num_masks: Option<usize>,
    pub iou: Option<f32>,
    pub apply_nms: Option<bool>,
    pub find_contours: bool,
    pub yolo_preds_format: Option<YOLOPredsFormat>,
    pub classes_excluded: Vec<usize>,
    pub classes_retained: Vec<usize>,
    pub min_width: Option<f32>,
    pub min_height: Option<f32>,
    pub db_unclip_ratio: Option<f32>,
    pub db_binary_thresh: Option<f32>,
    pub sam_kind: Option<SamKind>,
    pub sam_low_res_mask: Option<bool>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            class_names: vec![],
            keypoint_names: vec![],
            text_names: vec![],
            class_confs: vec![0.25f32],
            keypoint_confs: vec![0.3f32],
            text_confs: vec![0.25f32],
            apply_softmax: Some(false),
            num_classes: None,
            num_keypoints: None,
            num_masks: None,
            iou: None,
            find_contours: false,
            yolo_preds_format: None,
            classes_excluded: vec![],
            classes_retained: vec![],
            apply_nms: None,
            min_width: None,
            min_height: None,
            db_unclip_ratio: Some(1.5),
            db_binary_thresh: Some(0.2),
            sam_kind: None,
            sam_low_res_mask: None,
            topk: None,
            model: Default::default(),
            encoder: Default::default(),
            decoder: Default::default(),
            visual: Default::default(),
            textual: Default::default(),
            visual_encoder: Default::default(),
            textual_encoder: Default::default(),
            visual_decoder: Default::default(),
            textual_decoder: Default::default(),
            textual_decoder_merged: Default::default(),
            processor: ProcessorConfig::default(),
            size_encoder: Default::default(),
            size_decoder: Default::default(),
            coord_encoder: Default::default(),
            coord_decoder: Default::default(),
            visual_projection: Default::default(),
            textual_projection: Default::default(),
            version: None,
            task: None,
            scale: None,
            name: Default::default(),
        }
    }
}

impl ModelConfig {
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

    pub fn commit(mut self) -> anyhow::Result<Self> {
        // special case for yolo
        if self.name == "yolo" && self.model.file.is_empty() {
            // version-scale-task
            let mut y = String::new();
            if let Some(x) = self.version() {
                y.push_str(&x.to_string());
            }
            if let Some(x) = self.scale() {
                y.push_str(&format!("-{}", x));
            }
            if let Some(x) = self.task() {
                y.push_str(&format!("-{}", x.yolo_str()));
            }
            y.push_str(".onnx");
            self.model.file = y;
        }

        fn try_commit(name: &str, mut m: EngineConfig) -> anyhow::Result<EngineConfig> {
            if !m.file.is_empty() {
                m = m.try_commit(name)?;
                return Ok(m);
            }

            Ok(m)
        }

        self.model = try_commit(self.name, self.model)?;
        self.visual = try_commit(self.name, self.visual)?;
        self.textual = try_commit(self.name, self.textual)?;
        self.encoder = try_commit(self.name, self.encoder)?;
        self.decoder = try_commit(self.name, self.decoder)?;
        self.visual_encoder = try_commit(self.name, self.visual_encoder)?;
        self.textual_encoder = try_commit(self.name, self.textual_encoder)?;
        self.visual_decoder = try_commit(self.name, self.visual_decoder)?;
        self.textual_decoder = try_commit(self.name, self.textual_decoder)?;
        self.textual_decoder_merged = try_commit(self.name, self.textual_decoder_merged)?;
        self.size_encoder = try_commit(self.name, self.size_encoder)?;
        self.size_decoder = try_commit(self.name, self.size_decoder)?;
        self.coord_encoder = try_commit(self.name, self.coord_encoder)?;
        self.coord_decoder = try_commit(self.name, self.coord_decoder)?;
        self.visual_projection = try_commit(self.name, self.visual_projection)?;
        self.textual_projection = try_commit(self.name, self.textual_projection)?;

        Ok(self)
    }

    pub fn with_batch_size_all(mut self, batch_size: usize) -> Self {
        self.visual = self.visual.with_ixx(0, 0, batch_size.into());
        self.textual = self.textual.with_ixx(0, 0, batch_size.into());
        self.model = self.model.with_ixx(0, 0, batch_size.into());
        self.encoder = self.encoder.with_ixx(0, 0, batch_size.into());
        self.decoder = self.decoder.with_ixx(0, 0, batch_size.into());
        self.visual_encoder = self.visual_encoder.with_ixx(0, 0, batch_size.into());
        self.textual_encoder = self.textual_encoder.with_ixx(0, 0, batch_size.into());
        self.visual_decoder = self.visual_decoder.with_ixx(0, 0, batch_size.into());
        self.textual_decoder = self.textual_decoder.with_ixx(0, 0, batch_size.into());
        self.textual_decoder_merged = self
            .textual_decoder_merged
            .with_ixx(0, 0, batch_size.into());
        self.size_encoder = self.size_encoder.with_ixx(0, 0, batch_size.into());
        self.size_decoder = self.size_decoder.with_ixx(0, 0, batch_size.into());
        self.coord_encoder = self.coord_encoder.with_ixx(0, 0, batch_size.into());
        self.coord_decoder = self.coord_decoder.with_ixx(0, 0, batch_size.into());
        self.visual_projection = self.visual_projection.with_ixx(0, 0, batch_size.into());
        self.textual_projection = self.textual_projection.with_ixx(0, 0, batch_size.into());

        self
    }

    pub fn with_device_all(mut self, device: crate::Device) -> Self {
        self.visual = self.visual.with_device(device);
        self.textual = self.textual.with_device(device);
        self.model = self.model.with_device(device);
        self.encoder = self.encoder.with_device(device);
        self.decoder = self.decoder.with_device(device);
        self.visual_encoder = self.visual_encoder.with_device(device);
        self.textual_encoder = self.textual_encoder.with_device(device);
        self.visual_decoder = self.visual_decoder.with_device(device);
        self.textual_decoder = self.textual_decoder.with_device(device);
        self.textual_decoder_merged = self.textual_decoder_merged.with_device(device);
        self.size_encoder = self.size_encoder.with_device(device);
        self.size_decoder = self.size_decoder.with_device(device);
        self.coord_encoder = self.coord_encoder.with_device(device);
        self.coord_decoder = self.coord_decoder.with_device(device);
        self.visual_projection = self.visual_projection.with_device(device);
        self.textual_projection = self.textual_projection.with_device(device);

        self
    }

    pub fn with_dtype_all(mut self, dtype: crate::DType) -> Self {
        self.visual = self.visual.with_dtype(dtype);
        self.textual = self.textual.with_dtype(dtype);
        self.model = self.model.with_dtype(dtype);
        self.encoder = self.encoder.with_dtype(dtype);
        self.decoder = self.decoder.with_dtype(dtype);
        self.visual_encoder = self.visual_encoder.with_dtype(dtype);
        self.textual_encoder = self.textual_encoder.with_dtype(dtype);
        self.visual_decoder = self.visual_decoder.with_dtype(dtype);
        self.textual_decoder = self.textual_decoder.with_dtype(dtype);
        self.textual_decoder_merged = self.textual_decoder_merged.with_dtype(dtype);
        self.size_encoder = self.size_encoder.with_dtype(dtype);
        self.size_decoder = self.size_decoder.with_dtype(dtype);
        self.coord_encoder = self.coord_encoder.with_dtype(dtype);
        self.coord_decoder = self.coord_decoder.with_dtype(dtype);
        self.visual_projection = self.visual_projection.with_dtype(dtype);
        self.textual_projection = self.textual_projection.with_dtype(dtype);

        self
    }
}

impl_model_config_methods!(ModelConfig, model);
impl_model_config_methods!(ModelConfig, visual);
impl_model_config_methods!(ModelConfig, textual);
impl_model_config_methods!(ModelConfig, encoder);
impl_model_config_methods!(ModelConfig, decoder);
impl_model_config_methods!(ModelConfig, visual_encoder);
impl_model_config_methods!(ModelConfig, textual_encoder);
impl_model_config_methods!(ModelConfig, visual_decoder);
impl_model_config_methods!(ModelConfig, textual_decoder);
impl_model_config_methods!(ModelConfig, textual_decoder_merged);
impl_model_config_methods!(ModelConfig, size_encoder);
impl_model_config_methods!(ModelConfig, size_decoder);
impl_model_config_methods!(ModelConfig, coord_encoder);
impl_model_config_methods!(ModelConfig, coord_decoder);
impl_model_config_methods!(ModelConfig, visual_projection);
impl_model_config_methods!(ModelConfig, textual_projection);
impl_process_config_methods!(ModelConfig, processor);
