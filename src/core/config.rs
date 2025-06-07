use aksr::Builder;

use crate::{
    models::{SamKind, YOLOPredsFormat},
    ORTConfig, ProcessorConfig, Scale, Task, Version,
};

/// Configuration for model inference including engines, processors, and task settings.
#[derive(Builder, Debug, Clone)]
pub struct Config {
    // Basics
    pub name: &'static str,
    pub version: Option<Version>,
    pub task: Option<Task>,
    pub scale: Option<Scale>,

    // Engines
    pub model: ORTConfig,
    pub visual: ORTConfig,
    pub textual: ORTConfig,
    pub encoder: ORTConfig,
    pub decoder: ORTConfig,
    pub visual_encoder: ORTConfig,
    pub textual_encoder: ORTConfig,
    pub visual_decoder: ORTConfig,
    pub textual_decoder: ORTConfig,
    pub textual_decoder_merged: ORTConfig,
    pub size_encoder: ORTConfig,
    pub size_decoder: ORTConfig,
    pub coord_encoder: ORTConfig,
    pub coord_decoder: ORTConfig,
    pub visual_projection: ORTConfig,
    pub textual_projection: ORTConfig,

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

impl Default for Config {
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

impl Config {
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

        fn try_commit(name: &str, mut m: ORTConfig) -> anyhow::Result<ORTConfig> {
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

    pub fn with_num_dry_run_all(mut self, x: usize) -> Self {
        self.visual = self.visual.with_num_dry_run(x);
        self.textual = self.textual.with_num_dry_run(x);
        self.model = self.model.with_num_dry_run(x);
        self.encoder = self.encoder.with_num_dry_run(x);
        self.decoder = self.decoder.with_num_dry_run(x);
        self.visual_encoder = self.visual_encoder.with_num_dry_run(x);
        self.textual_encoder = self.textual_encoder.with_num_dry_run(x);
        self.visual_decoder = self.visual_decoder.with_num_dry_run(x);
        self.textual_decoder = self.textual_decoder.with_num_dry_run(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_num_dry_run(x);
        self.size_encoder = self.size_encoder.with_num_dry_run(x);
        self.size_decoder = self.size_decoder.with_num_dry_run(x);
        self.coord_encoder = self.coord_encoder.with_num_dry_run(x);
        self.coord_decoder = self.coord_decoder.with_num_dry_run(x);
        self.visual_projection = self.visual_projection.with_num_dry_run(x);
        self.textual_projection = self.textual_projection.with_num_dry_run(x);

        self
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

    pub fn with_graph_opt_level_all(mut self, level: u8) -> Self {
        self.visual = self.visual.with_graph_opt_level(level);
        self.textual = self.textual.with_graph_opt_level(level);
        self.model = self.model.with_graph_opt_level(level);
        self.encoder = self.encoder.with_graph_opt_level(level);
        self.decoder = self.decoder.with_graph_opt_level(level);
        self.visual_encoder = self.visual_encoder.with_graph_opt_level(level);
        self.textual_encoder = self.textual_encoder.with_graph_opt_level(level);
        self.visual_decoder = self.visual_decoder.with_graph_opt_level(level);
        self.textual_decoder = self.textual_decoder.with_graph_opt_level(level);
        self.textual_decoder_merged = self.textual_decoder_merged.with_graph_opt_level(level);
        self.size_encoder = self.size_encoder.with_graph_opt_level(level);
        self.size_decoder = self.size_decoder.with_graph_opt_level(level);
        self.coord_encoder = self.coord_encoder.with_graph_opt_level(level);
        self.coord_decoder = self.coord_decoder.with_graph_opt_level(level);
        self.visual_projection = self.visual_projection.with_graph_opt_level(level);
        self.textual_projection = self.textual_projection.with_graph_opt_level(level);
        self
    }

    pub fn with_num_intra_threads_all(mut self, num_threads: usize) -> Self {
        self.visual = self.visual.with_num_intra_threads(num_threads);
        self.textual = self.textual.with_num_intra_threads(num_threads);
        self.model = self.model.with_num_intra_threads(num_threads);
        self.encoder = self.encoder.with_num_intra_threads(num_threads);
        self.decoder = self.decoder.with_num_intra_threads(num_threads);
        self.visual_encoder = self.visual_encoder.with_num_intra_threads(num_threads);
        self.textual_encoder = self.textual_encoder.with_num_intra_threads(num_threads);
        self.visual_decoder = self.visual_decoder.with_num_intra_threads(num_threads);
        self.textual_decoder = self.textual_decoder.with_num_intra_threads(num_threads);
        self.textual_decoder_merged = self
            .textual_decoder_merged
            .with_num_intra_threads(num_threads);
        self.size_encoder = self.size_encoder.with_num_intra_threads(num_threads);
        self.size_decoder = self.size_decoder.with_num_intra_threads(num_threads);
        self.coord_encoder = self.coord_encoder.with_num_intra_threads(num_threads);
        self.coord_decoder = self.coord_decoder.with_num_intra_threads(num_threads);
        self.visual_projection = self.visual_projection.with_num_intra_threads(num_threads);
        self.textual_projection = self.textual_projection.with_num_intra_threads(num_threads);
        self
    }

    pub fn with_num_inter_threads_all(mut self, num_threads: usize) -> Self {
        self.visual = self.visual.with_num_inter_threads(num_threads);
        self.textual = self.textual.with_num_inter_threads(num_threads);
        self.model = self.model.with_num_inter_threads(num_threads);
        self.encoder = self.encoder.with_num_inter_threads(num_threads);
        self.decoder = self.decoder.with_num_inter_threads(num_threads);
        self.visual_encoder = self.visual_encoder.with_num_inter_threads(num_threads);
        self.textual_encoder = self.textual_encoder.with_num_inter_threads(num_threads);
        self.visual_decoder = self.visual_decoder.with_num_inter_threads(num_threads);
        self.textual_decoder = self.textual_decoder.with_num_inter_threads(num_threads);
        self.textual_decoder_merged = self
            .textual_decoder_merged
            .with_num_inter_threads(num_threads);
        self.size_encoder = self.size_encoder.with_num_inter_threads(num_threads);
        self.size_decoder = self.size_decoder.with_num_inter_threads(num_threads);
        self.coord_encoder = self.coord_encoder.with_num_inter_threads(num_threads);
        self.coord_decoder = self.coord_decoder.with_num_inter_threads(num_threads);
        self.visual_projection = self.visual_projection.with_num_inter_threads(num_threads);
        self.textual_projection = self.textual_projection.with_num_inter_threads(num_threads);
        self
    }

    pub fn with_cpu_arena_allocator_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_cpu_arena_allocator(x);
        self.textual = self.textual.with_cpu_arena_allocator(x);
        self.model = self.model.with_cpu_arena_allocator(x);
        self.encoder = self.encoder.with_cpu_arena_allocator(x);
        self.decoder = self.decoder.with_cpu_arena_allocator(x);
        self.visual_encoder = self.visual_encoder.with_cpu_arena_allocator(x);
        self.textual_encoder = self.textual_encoder.with_cpu_arena_allocator(x);
        self.visual_decoder = self.visual_decoder.with_cpu_arena_allocator(x);
        self.textual_decoder = self.textual_decoder.with_cpu_arena_allocator(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_cpu_arena_allocator(x);
        self.size_encoder = self.size_encoder.with_cpu_arena_allocator(x);
        self.size_decoder = self.size_decoder.with_cpu_arena_allocator(x);
        self.coord_encoder = self.coord_encoder.with_cpu_arena_allocator(x);
        self.coord_decoder = self.coord_decoder.with_cpu_arena_allocator(x);
        self.visual_projection = self.visual_projection.with_cpu_arena_allocator(x);
        self.textual_projection = self.textual_projection.with_cpu_arena_allocator(x);
        self
    }

    pub fn with_openvino_dynamic_shapes_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_openvino_dynamic_shapes(x);
        self.textual = self.textual.with_openvino_dynamic_shapes(x);
        self.model = self.model.with_openvino_dynamic_shapes(x);
        self.encoder = self.encoder.with_openvino_dynamic_shapes(x);
        self.decoder = self.decoder.with_openvino_dynamic_shapes(x);
        self.visual_encoder = self.visual_encoder.with_openvino_dynamic_shapes(x);
        self.textual_encoder = self.textual_encoder.with_openvino_dynamic_shapes(x);
        self.visual_decoder = self.visual_decoder.with_openvino_dynamic_shapes(x);
        self.textual_decoder = self.textual_decoder.with_openvino_dynamic_shapes(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_openvino_dynamic_shapes(x);
        self.size_encoder = self.size_encoder.with_openvino_dynamic_shapes(x);
        self.size_decoder = self.size_decoder.with_openvino_dynamic_shapes(x);
        self.coord_encoder = self.coord_encoder.with_openvino_dynamic_shapes(x);
        self.coord_decoder = self.coord_decoder.with_openvino_dynamic_shapes(x);
        self.visual_projection = self.visual_projection.with_openvino_dynamic_shapes(x);
        self.textual_projection = self.textual_projection.with_openvino_dynamic_shapes(x);
        self
    }

    pub fn with_openvino_opencl_throttling_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_openvino_opencl_throttling(x);
        self.textual = self.textual.with_openvino_opencl_throttling(x);
        self.model = self.model.with_openvino_opencl_throttling(x);
        self.encoder = self.encoder.with_openvino_opencl_throttling(x);
        self.decoder = self.decoder.with_openvino_opencl_throttling(x);
        self.visual_encoder = self.visual_encoder.with_openvino_opencl_throttling(x);
        self.textual_encoder = self.textual_encoder.with_openvino_opencl_throttling(x);
        self.visual_decoder = self.visual_decoder.with_openvino_opencl_throttling(x);
        self.textual_decoder = self.textual_decoder.with_openvino_opencl_throttling(x);
        self.textual_decoder_merged = self
            .textual_decoder_merged
            .with_openvino_opencl_throttling(x);
        self.size_encoder = self.size_encoder.with_openvino_opencl_throttling(x);
        self.size_decoder = self.size_decoder.with_openvino_opencl_throttling(x);
        self.coord_encoder = self.coord_encoder.with_openvino_opencl_throttling(x);
        self.coord_decoder = self.coord_decoder.with_openvino_opencl_throttling(x);
        self.visual_projection = self.visual_projection.with_openvino_opencl_throttling(x);
        self.textual_projection = self.textual_projection.with_openvino_opencl_throttling(x);
        self
    }

    pub fn with_openvino_qdq_optimizer_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_openvino_qdq_optimizer(x);
        self.textual = self.textual.with_openvino_qdq_optimizer(x);
        self.model = self.model.with_openvino_qdq_optimizer(x);
        self.encoder = self.encoder.with_openvino_qdq_optimizer(x);
        self.decoder = self.decoder.with_openvino_qdq_optimizer(x);
        self.visual_encoder = self.visual_encoder.with_openvino_qdq_optimizer(x);
        self.textual_encoder = self.textual_encoder.with_openvino_qdq_optimizer(x);
        self.visual_decoder = self.visual_decoder.with_openvino_qdq_optimizer(x);
        self.textual_decoder = self.textual_decoder.with_openvino_qdq_optimizer(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_openvino_qdq_optimizer(x);
        self.size_encoder = self.size_encoder.with_openvino_qdq_optimizer(x);
        self.size_decoder = self.size_decoder.with_openvino_qdq_optimizer(x);
        self.coord_encoder = self.coord_encoder.with_openvino_qdq_optimizer(x);
        self.coord_decoder = self.coord_decoder.with_openvino_qdq_optimizer(x);
        self.visual_projection = self.visual_projection.with_openvino_qdq_optimizer(x);
        self.textual_projection = self.textual_projection.with_openvino_qdq_optimizer(x);
        self
    }

    pub fn with_openvino_num_threads_all(mut self, num_threads: usize) -> Self {
        self.visual = self.visual.with_openvino_num_threads(num_threads);
        self.textual = self.textual.with_openvino_num_threads(num_threads);
        self.model = self.model.with_openvino_num_threads(num_threads);
        self.encoder = self.encoder.with_openvino_num_threads(num_threads);
        self.decoder = self.decoder.with_openvino_num_threads(num_threads);
        self.visual_encoder = self.visual_encoder.with_openvino_num_threads(num_threads);
        self.textual_encoder = self.textual_encoder.with_openvino_num_threads(num_threads);
        self.visual_decoder = self.visual_decoder.with_openvino_num_threads(num_threads);
        self.textual_decoder = self.textual_decoder.with_openvino_num_threads(num_threads);
        self.textual_decoder_merged = self
            .textual_decoder_merged
            .with_openvino_num_threads(num_threads);
        self.size_encoder = self.size_encoder.with_openvino_num_threads(num_threads);
        self.size_decoder = self.size_decoder.with_openvino_num_threads(num_threads);
        self.coord_encoder = self.coord_encoder.with_openvino_num_threads(num_threads);
        self.coord_decoder = self.coord_decoder.with_openvino_num_threads(num_threads);
        self.visual_projection = self
            .visual_projection
            .with_openvino_num_threads(num_threads);
        self.textual_projection = self
            .textual_projection
            .with_openvino_num_threads(num_threads);
        self
    }

    // onednn
    pub fn with_onednn_arena_allocator_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_onednn_arena_allocator(x);
        self.textual = self.textual.with_onednn_arena_allocator(x);
        self.model = self.model.with_onednn_arena_allocator(x);
        self.encoder = self.encoder.with_onednn_arena_allocator(x);
        self.decoder = self.decoder.with_onednn_arena_allocator(x);
        self.visual_encoder = self.visual_encoder.with_onednn_arena_allocator(x);
        self.textual_encoder = self.textual_encoder.with_onednn_arena_allocator(x);
        self.visual_decoder = self.visual_decoder.with_onednn_arena_allocator(x);
        self.textual_decoder = self.textual_decoder.with_onednn_arena_allocator(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_onednn_arena_allocator(x);
        self.size_encoder = self.size_encoder.with_onednn_arena_allocator(x);
        self.size_decoder = self.size_decoder.with_onednn_arena_allocator(x);
        self.coord_encoder = self.coord_encoder.with_onednn_arena_allocator(x);
        self.coord_decoder = self.coord_decoder.with_onednn_arena_allocator(x);
        self.visual_projection = self.visual_projection.with_onednn_arena_allocator(x);
        self.textual_projection = self.textual_projection.with_onednn_arena_allocator(x);
        self
    }

    // tensorrt
    pub fn with_tensorrt_fp16_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_tensorrt_fp16(x);
        self.textual = self.textual.with_tensorrt_fp16(x);
        self.model = self.model.with_tensorrt_fp16(x);
        self.encoder = self.encoder.with_tensorrt_fp16(x);
        self.decoder = self.decoder.with_tensorrt_fp16(x);
        self.visual_encoder = self.visual_encoder.with_tensorrt_fp16(x);
        self.textual_encoder = self.textual_encoder.with_tensorrt_fp16(x);
        self.visual_decoder = self.visual_decoder.with_tensorrt_fp16(x);
        self.textual_decoder = self.textual_decoder.with_tensorrt_fp16(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_tensorrt_fp16(x);
        self.size_encoder = self.size_encoder.with_tensorrt_fp16(x);
        self.size_decoder = self.size_decoder.with_tensorrt_fp16(x);
        self.coord_encoder = self.coord_encoder.with_tensorrt_fp16(x);
        self.coord_decoder = self.coord_decoder.with_tensorrt_fp16(x);
        self.visual_projection = self.visual_projection.with_tensorrt_fp16(x);
        self.textual_projection = self.textual_projection.with_tensorrt_fp16(x);
        self
    }

    pub fn with_tensorrt_engine_cache_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_tensorrt_engine_cache(x);
        self.textual = self.textual.with_tensorrt_engine_cache(x);
        self.model = self.model.with_tensorrt_engine_cache(x);
        self.encoder = self.encoder.with_tensorrt_engine_cache(x);
        self.decoder = self.decoder.with_tensorrt_engine_cache(x);
        self.visual_encoder = self.visual_encoder.with_tensorrt_engine_cache(x);
        self.textual_encoder = self.textual_encoder.with_tensorrt_engine_cache(x);
        self.visual_decoder = self.visual_decoder.with_tensorrt_engine_cache(x);
        self.textual_decoder = self.textual_decoder.with_tensorrt_engine_cache(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_tensorrt_engine_cache(x);
        self.size_encoder = self.size_encoder.with_tensorrt_engine_cache(x);
        self.size_decoder = self.size_decoder.with_tensorrt_engine_cache(x);
        self.coord_encoder = self.coord_encoder.with_tensorrt_engine_cache(x);
        self.coord_decoder = self.coord_decoder.with_tensorrt_engine_cache(x);
        self.visual_projection = self.visual_projection.with_tensorrt_engine_cache(x);
        self.textual_projection = self.textual_projection.with_tensorrt_engine_cache(x);
        self
    }

    pub fn with_tensorrt_timing_cache_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_tensorrt_timing_cache(x);
        self.textual = self.textual.with_tensorrt_timing_cache(x);
        self.model = self.model.with_tensorrt_timing_cache(x);
        self.encoder = self.encoder.with_tensorrt_timing_cache(x);
        self.decoder = self.decoder.with_tensorrt_timing_cache(x);
        self.visual_encoder = self.visual_encoder.with_tensorrt_timing_cache(x);
        self.textual_encoder = self.textual_encoder.with_tensorrt_timing_cache(x);
        self.visual_decoder = self.visual_decoder.with_tensorrt_timing_cache(x);
        self.textual_decoder = self.textual_decoder.with_tensorrt_timing_cache(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_tensorrt_timing_cache(x);
        self.size_encoder = self.size_encoder.with_tensorrt_timing_cache(x);
        self.size_decoder = self.size_decoder.with_tensorrt_timing_cache(x);
        self.coord_encoder = self.coord_encoder.with_tensorrt_timing_cache(x);
        self.coord_decoder = self.coord_decoder.with_tensorrt_timing_cache(x);
        self.visual_projection = self.visual_projection.with_tensorrt_timing_cache(x);
        self.textual_projection = self.textual_projection.with_tensorrt_timing_cache(x);
        self
    }

    // coreml
    pub fn with_coreml_static_input_shapes_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_coreml_static_input_shapes(x);
        self.textual = self.textual.with_coreml_static_input_shapes(x);
        self.model = self.model.with_coreml_static_input_shapes(x);
        self.encoder = self.encoder.with_coreml_static_input_shapes(x);
        self.decoder = self.decoder.with_coreml_static_input_shapes(x);
        self.visual_encoder = self.visual_encoder.with_coreml_static_input_shapes(x);
        self.textual_encoder = self.textual_encoder.with_coreml_static_input_shapes(x);
        self.visual_decoder = self.visual_decoder.with_coreml_static_input_shapes(x);
        self.textual_decoder = self.textual_decoder.with_coreml_static_input_shapes(x);
        self.textual_decoder_merged = self
            .textual_decoder_merged
            .with_coreml_static_input_shapes(x);
        self.size_encoder = self.size_encoder.with_coreml_static_input_shapes(x);
        self.size_decoder = self.size_decoder.with_coreml_static_input_shapes(x);
        self.coord_encoder = self.coord_encoder.with_coreml_static_input_shapes(x);
        self.coord_decoder = self.coord_decoder.with_coreml_static_input_shapes(x);
        self.visual_projection = self.visual_projection.with_coreml_static_input_shapes(x);
        self.textual_projection = self.textual_projection.with_coreml_static_input_shapes(x);
        self
    }

    pub fn with_coreml_subgraph_running_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_coreml_subgraph_running(x);
        self.textual = self.textual.with_coreml_subgraph_running(x);
        self.model = self.model.with_coreml_subgraph_running(x);
        self.encoder = self.encoder.with_coreml_subgraph_running(x);
        self.decoder = self.decoder.with_coreml_subgraph_running(x);
        self.visual_encoder = self.visual_encoder.with_coreml_subgraph_running(x);
        self.textual_encoder = self.textual_encoder.with_coreml_subgraph_running(x);
        self.visual_decoder = self.visual_decoder.with_coreml_subgraph_running(x);
        self.textual_decoder = self.textual_decoder.with_coreml_subgraph_running(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_coreml_subgraph_running(x);
        self.size_encoder = self.size_encoder.with_coreml_subgraph_running(x);
        self.size_decoder = self.size_decoder.with_coreml_subgraph_running(x);
        self.coord_encoder = self.coord_encoder.with_coreml_subgraph_running(x);
        self.coord_decoder = self.coord_decoder.with_coreml_subgraph_running(x);
        self.visual_projection = self.visual_projection.with_coreml_subgraph_running(x);
        self.textual_projection = self.textual_projection.with_coreml_subgraph_running(x);
        self
    }

    // cann
    pub fn with_cann_graph_inference_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_cann_graph_inference(x);
        self.textual = self.textual.with_cann_graph_inference(x);
        self.model = self.model.with_cann_graph_inference(x);
        self.encoder = self.encoder.with_cann_graph_inference(x);
        self.decoder = self.decoder.with_cann_graph_inference(x);
        self.visual_encoder = self.visual_encoder.with_cann_graph_inference(x);
        self.textual_encoder = self.textual_encoder.with_cann_graph_inference(x);
        self.visual_decoder = self.visual_decoder.with_cann_graph_inference(x);
        self.textual_decoder = self.textual_decoder.with_cann_graph_inference(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_cann_graph_inference(x);
        self.size_encoder = self.size_encoder.with_cann_graph_inference(x);
        self.size_decoder = self.size_decoder.with_cann_graph_inference(x);
        self.coord_encoder = self.coord_encoder.with_cann_graph_inference(x);
        self.coord_decoder = self.coord_decoder.with_cann_graph_inference(x);
        self.visual_projection = self.visual_projection.with_cann_graph_inference(x);
        self.textual_projection = self.textual_projection.with_cann_graph_inference(x);
        self
    }

    pub fn with_cann_dump_graphs_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_cann_dump_graphs(x);
        self.textual = self.textual.with_cann_dump_graphs(x);
        self.model = self.model.with_cann_dump_graphs(x);
        self.encoder = self.encoder.with_cann_dump_graphs(x);
        self.decoder = self.decoder.with_cann_dump_graphs(x);
        self.visual_encoder = self.visual_encoder.with_cann_dump_graphs(x);
        self.textual_encoder = self.textual_encoder.with_cann_dump_graphs(x);
        self.visual_decoder = self.visual_decoder.with_cann_dump_graphs(x);
        self.textual_decoder = self.textual_decoder.with_cann_dump_graphs(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_cann_dump_graphs(x);
        self.size_encoder = self.size_encoder.with_cann_dump_graphs(x);
        self.size_decoder = self.size_decoder.with_cann_dump_graphs(x);
        self.coord_encoder = self.coord_encoder.with_cann_dump_graphs(x);
        self.coord_decoder = self.coord_decoder.with_cann_dump_graphs(x);
        self.visual_projection = self.visual_projection.with_cann_dump_graphs(x);
        self.textual_projection = self.textual_projection.with_cann_dump_graphs(x);
        self
    }

    pub fn with_cann_dump_om_model_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_cann_dump_om_model(x);
        self.textual = self.textual.with_cann_dump_om_model(x);
        self.model = self.model.with_cann_dump_om_model(x);
        self.encoder = self.encoder.with_cann_dump_om_model(x);
        self.decoder = self.decoder.with_cann_dump_om_model(x);
        self.visual_encoder = self.visual_encoder.with_cann_dump_om_model(x);
        self.textual_encoder = self.textual_encoder.with_cann_dump_om_model(x);
        self.visual_decoder = self.visual_decoder.with_cann_dump_om_model(x);
        self.textual_decoder = self.textual_decoder.with_cann_dump_om_model(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_cann_dump_om_model(x);
        self.size_encoder = self.size_encoder.with_cann_dump_om_model(x);
        self.size_decoder = self.size_decoder.with_cann_dump_om_model(x);
        self.coord_encoder = self.coord_encoder.with_cann_dump_om_model(x);
        self.coord_decoder = self.coord_decoder.with_cann_dump_om_model(x);
        self.visual_projection = self.visual_projection.with_cann_dump_om_model(x);
        self.textual_projection = self.textual_projection.with_cann_dump_om_model(x);
        self
    }

    // nnapi
    pub fn with_nnapi_cpu_only_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_nnapi_cpu_only(x);
        self.textual = self.textual.with_nnapi_cpu_only(x);
        self.model = self.model.with_nnapi_cpu_only(x);
        self.encoder = self.encoder.with_nnapi_cpu_only(x);
        self.decoder = self.decoder.with_nnapi_cpu_only(x);
        self.visual_encoder = self.visual_encoder.with_nnapi_cpu_only(x);
        self.textual_encoder = self.textual_encoder.with_nnapi_cpu_only(x);
        self.visual_decoder = self.visual_decoder.with_nnapi_cpu_only(x);
        self.textual_decoder = self.textual_decoder.with_nnapi_cpu_only(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_nnapi_cpu_only(x);
        self.size_encoder = self.size_encoder.with_nnapi_cpu_only(x);
        self.size_decoder = self.size_decoder.with_nnapi_cpu_only(x);
        self.coord_encoder = self.coord_encoder.with_nnapi_cpu_only(x);
        self.coord_decoder = self.coord_decoder.with_nnapi_cpu_only(x);
        self.visual_projection = self.visual_projection.with_nnapi_cpu_only(x);
        self.textual_projection = self.textual_projection.with_nnapi_cpu_only(x);
        self
    }

    pub fn with_nnapi_disable_cpu_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_nnapi_disable_cpu(x);
        self.textual = self.textual.with_nnapi_disable_cpu(x);
        self.model = self.model.with_nnapi_disable_cpu(x);
        self.encoder = self.encoder.with_nnapi_disable_cpu(x);
        self.decoder = self.decoder.with_nnapi_disable_cpu(x);
        self.visual_encoder = self.visual_encoder.with_nnapi_disable_cpu(x);
        self.textual_encoder = self.textual_encoder.with_nnapi_disable_cpu(x);
        self.visual_decoder = self.visual_decoder.with_nnapi_disable_cpu(x);
        self.textual_decoder = self.textual_decoder.with_nnapi_disable_cpu(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_nnapi_disable_cpu(x);
        self.size_encoder = self.size_encoder.with_nnapi_disable_cpu(x);
        self.size_decoder = self.size_decoder.with_nnapi_disable_cpu(x);
        self.coord_encoder = self.coord_encoder.with_nnapi_disable_cpu(x);
        self.coord_decoder = self.coord_decoder.with_nnapi_disable_cpu(x);
        self.visual_projection = self.visual_projection.with_nnapi_disable_cpu(x);
        self.textual_projection = self.textual_projection.with_nnapi_disable_cpu(x);
        self
    }

    pub fn with_nnapi_fp16_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_nnapi_fp16(x);
        self.textual = self.textual.with_nnapi_fp16(x);
        self.model = self.model.with_nnapi_fp16(x);
        self.encoder = self.encoder.with_nnapi_fp16(x);
        self.decoder = self.decoder.with_nnapi_fp16(x);
        self.visual_encoder = self.visual_encoder.with_nnapi_fp16(x);
        self.textual_encoder = self.textual_encoder.with_nnapi_fp16(x);
        self.visual_decoder = self.visual_decoder.with_nnapi_fp16(x);
        self.textual_decoder = self.textual_decoder.with_nnapi_fp16(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_nnapi_fp16(x);
        self.size_encoder = self.size_encoder.with_nnapi_fp16(x);
        self.size_decoder = self.size_decoder.with_nnapi_fp16(x);
        self.coord_encoder = self.coord_encoder.with_nnapi_fp16(x);
        self.coord_decoder = self.coord_decoder.with_nnapi_fp16(x);
        self.visual_projection = self.visual_projection.with_nnapi_fp16(x);
        self.textual_projection = self.textual_projection.with_nnapi_fp16(x);
        self
    }

    pub fn with_nnapi_nchw_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_nnapi_nchw(x);
        self.textual = self.textual.with_nnapi_nchw(x);
        self.model = self.model.with_nnapi_nchw(x);
        self.encoder = self.encoder.with_nnapi_nchw(x);
        self.decoder = self.decoder.with_nnapi_nchw(x);
        self.visual_encoder = self.visual_encoder.with_nnapi_nchw(x);
        self.textual_encoder = self.textual_encoder.with_nnapi_nchw(x);
        self.visual_decoder = self.visual_decoder.with_nnapi_nchw(x);
        self.textual_decoder = self.textual_decoder.with_nnapi_nchw(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_nnapi_nchw(x);
        self.size_encoder = self.size_encoder.with_nnapi_nchw(x);
        self.size_decoder = self.size_decoder.with_nnapi_nchw(x);
        self.coord_encoder = self.coord_encoder.with_nnapi_nchw(x);
        self.coord_decoder = self.coord_decoder.with_nnapi_nchw(x);
        self.visual_projection = self.visual_projection.with_nnapi_nchw(x);
        self.textual_projection = self.textual_projection.with_nnapi_nchw(x);
        self
    }

    // armnn
    pub fn with_armnn_arena_allocator_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_armnn_arena_allocator(x);
        self.textual = self.textual.with_armnn_arena_allocator(x);
        self.model = self.model.with_armnn_arena_allocator(x);
        self.encoder = self.encoder.with_armnn_arena_allocator(x);
        self.decoder = self.decoder.with_armnn_arena_allocator(x);
        self.visual_encoder = self.visual_encoder.with_armnn_arena_allocator(x);
        self.textual_encoder = self.textual_encoder.with_armnn_arena_allocator(x);
        self.visual_decoder = self.visual_decoder.with_armnn_arena_allocator(x);
        self.textual_decoder = self.textual_decoder.with_armnn_arena_allocator(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_armnn_arena_allocator(x);
        self.size_encoder = self.size_encoder.with_armnn_arena_allocator(x);
        self.size_decoder = self.size_decoder.with_armnn_arena_allocator(x);
        self.coord_encoder = self.coord_encoder.with_armnn_arena_allocator(x);
        self.coord_decoder = self.coord_decoder.with_armnn_arena_allocator(x);
        self.visual_projection = self.visual_projection.with_armnn_arena_allocator(x);
        self.textual_projection = self.textual_projection.with_armnn_arena_allocator(x);
        self
    }

    // migraphx
    pub fn with_migraphx_fp16_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_migraphx_fp16(x);
        self.textual = self.textual.with_migraphx_fp16(x);
        self.model = self.model.with_migraphx_fp16(x);
        self.encoder = self.encoder.with_migraphx_fp16(x);
        self.decoder = self.decoder.with_migraphx_fp16(x);
        self.visual_encoder = self.visual_encoder.with_migraphx_fp16(x);
        self.textual_encoder = self.textual_encoder.with_migraphx_fp16(x);
        self.visual_decoder = self.visual_decoder.with_migraphx_fp16(x);
        self.textual_decoder = self.textual_decoder.with_migraphx_fp16(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_migraphx_fp16(x);
        self.size_encoder = self.size_encoder.with_migraphx_fp16(x);
        self.size_decoder = self.size_decoder.with_migraphx_fp16(x);
        self.coord_encoder = self.coord_encoder.with_migraphx_fp16(x);
        self.coord_decoder = self.coord_decoder.with_migraphx_fp16(x);
        self.visual_projection = self.visual_projection.with_migraphx_fp16(x);
        self.textual_projection = self.textual_projection.with_migraphx_fp16(x);
        self
    }

    pub fn with_migraphx_exhaustive_tune_all(mut self, x: bool) -> Self {
        self.visual = self.visual.with_migraphx_exhaustive_tune(x);
        self.textual = self.textual.with_migraphx_exhaustive_tune(x);
        self.model = self.model.with_migraphx_exhaustive_tune(x);
        self.encoder = self.encoder.with_migraphx_exhaustive_tune(x);
        self.decoder = self.decoder.with_migraphx_exhaustive_tune(x);
        self.visual_encoder = self.visual_encoder.with_migraphx_exhaustive_tune(x);
        self.textual_encoder = self.textual_encoder.with_migraphx_exhaustive_tune(x);
        self.visual_decoder = self.visual_decoder.with_migraphx_exhaustive_tune(x);
        self.textual_decoder = self.textual_decoder.with_migraphx_exhaustive_tune(x);
        self.textual_decoder_merged = self.textual_decoder_merged.with_migraphx_exhaustive_tune(x);
        self.size_encoder = self.size_encoder.with_migraphx_exhaustive_tune(x);
        self.size_decoder = self.size_decoder.with_migraphx_exhaustive_tune(x);
        self.coord_encoder = self.coord_encoder.with_migraphx_exhaustive_tune(x);
        self.coord_decoder = self.coord_decoder.with_migraphx_exhaustive_tune(x);
        self.visual_projection = self.visual_projection.with_migraphx_exhaustive_tune(x);
        self.textual_projection = self.textual_projection.with_migraphx_exhaustive_tune(x);
        self
    }
}

impl_ort_config_methods!(Config, model);
impl_ort_config_methods!(Config, visual);
impl_ort_config_methods!(Config, textual);
impl_ort_config_methods!(Config, encoder);
impl_ort_config_methods!(Config, decoder);
impl_ort_config_methods!(Config, visual_encoder);
impl_ort_config_methods!(Config, textual_encoder);
impl_ort_config_methods!(Config, visual_decoder);
impl_ort_config_methods!(Config, textual_decoder);
impl_ort_config_methods!(Config, textual_decoder_merged);
impl_ort_config_methods!(Config, size_encoder);
impl_ort_config_methods!(Config, size_decoder);
impl_ort_config_methods!(Config, coord_encoder);
impl_ort_config_methods!(Config, coord_decoder);
impl_ort_config_methods!(Config, visual_projection);
impl_ort_config_methods!(Config, textual_projection);
impl_processor_config_methods!(Config, processor);
