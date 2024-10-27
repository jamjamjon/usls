//! Options for build models.

use aksr::Builder;
use anyhow::Result;

use crate::{
    models::{SamKind, SapiensTask, YOLOPreds, YOLOTask, YOLOVersion},
    Device, Hub, Iiix, MinOptMax, Task,
};

/// Options for building models
#[derive(Builder, Debug, Clone)]
pub struct Options {
    pub onnx_path: String,
    pub task: Task,
    pub device: Device,
    pub batch_size: usize,
    pub iiixs: Vec<Iiix>,
    pub profile: bool,
    pub num_dry_run: usize,

    // trt related
    pub trt_engine_cache_enable: bool,
    pub trt_int8: bool,
    pub trt_fp16: bool,

    // options for Vision and Language models
    pub nc: Option<usize>,
    pub nk: Option<usize>,
    pub nm: Option<usize>,
    pub confs: Vec<f32>,
    pub confs2: Vec<f32>,
    pub confs3: Vec<f32>,
    pub kconfs: Vec<f32>,
    pub iou: Option<f32>,
    #[args(setter = false)]
    pub tokenizer: Option<String>,
    #[args(setter = false)]
    pub vocab: Option<String>,
    pub context_length: Option<usize>,
    pub names: Option<Vec<String>>,  // names
    pub names2: Option<Vec<String>>, // names2
    pub names3: Option<Vec<String>>, // names3
    pub min_width: Option<f32>,
    pub min_height: Option<f32>,
    pub unclip_ratio: f32, // DB
    pub yolo_task: Option<YOLOTask>,
    pub yolo_version: Option<YOLOVersion>,
    pub yolo_preds: Option<YOLOPreds>,
    pub find_contours: bool,
    pub sam_kind: Option<SamKind>,
    pub low_res_mask: Option<bool>,
    pub sapiens_task: Option<SapiensTask>,
    pub classes_excluded: Vec<isize>,
    pub classes_retained: Vec<isize>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            onnx_path: String::new(),
            device: Device::Cuda(0),
            profile: false,
            batch_size: 1,
            iiixs: vec![],
            num_dry_run: 3,

            trt_engine_cache_enable: true,
            trt_int8: false,
            trt_fp16: false,
            nc: None,
            nk: None,
            nm: None,
            confs: vec![0.3f32],
            confs2: vec![0.3f32],
            confs3: vec![0.3f32],
            kconfs: vec![0.5f32],
            iou: None,
            tokenizer: None,
            vocab: None,
            context_length: None,
            names: None,
            names2: None,
            names3: None,
            min_width: None,
            min_height: None,
            unclip_ratio: 1.5,
            yolo_task: None,
            yolo_version: None,
            yolo_preds: None,
            find_contours: false,
            sam_kind: None,
            low_res_mask: None,
            sapiens_task: None,
            task: Task::Untitled,
            classes_excluded: vec![],
            classes_retained: vec![],
        }
    }
}

impl Options {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_model(mut self, onnx_path: &str) -> Result<Self> {
        self.onnx_path = Hub::new()?.fetch(onnx_path)?.commit()?;
        Ok(self)
    }

    pub fn with_batch(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    pub fn with_cuda(mut self, id: usize) -> Self {
        self.device = Device::Cuda(id);
        self
    }

    pub fn with_trt(mut self, id: usize) -> Self {
        self.device = Device::Trt(id);
        self
    }

    pub fn with_cpu(mut self) -> Self {
        self.device = Device::Cpu(0);
        self
    }

    pub fn with_coreml(mut self, id: usize) -> Self {
        self.device = Device::CoreML(id);
        self
    }

    pub fn with_vocab(mut self, vocab: &str) -> Result<Self> {
        self.vocab = Some(Hub::new()?.fetch(vocab)?.commit()?);
        Ok(self)
    }

    pub fn with_tokenizer(mut self, tokenizer: &str) -> Result<Self> {
        self.tokenizer = Some(Hub::new()?.fetch(tokenizer)?.commit()?);
        Ok(self)
    }

    pub fn with_ixx(mut self, i: usize, ii: usize, x: MinOptMax) -> Self {
        self.iiixs.push(Iiix::from((i, ii, x)));
        self
    }

    pub fn exclude_classes(mut self, xs: &[isize]) -> Self {
        self.classes_retained.clear();
        self.classes_excluded.extend_from_slice(xs);
        self
    }

    pub fn retain_classes(mut self, xs: &[isize]) -> Self {
        self.classes_excluded.clear();
        self.classes_retained.extend_from_slice(xs);
        self
    }
}
