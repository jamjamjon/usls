//! Options for build models.

use anyhow::Result;

use crate::{
    models::{SamKind, SapiensTask, YOLOPreds, YOLOTask, YOLOVersion},
    Device, Hub, Iiix, MinOptMax, Task,
};

/// Options for building models
#[derive(Debug, Clone)]
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
    pub trt_int8_enable: bool,
    pub trt_fp16_enable: bool,

    // options for Vision and Language models
    pub nc: Option<usize>,
    pub nk: Option<usize>,
    pub nm: Option<usize>,
    pub confs: Vec<f32>,
    pub confs2: Vec<f32>,
    pub confs3: Vec<f32>,
    pub kconfs: Vec<f32>,
    pub iou: Option<f32>,
    pub tokenizer: Option<String>,
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
    pub use_low_res_mask: Option<bool>,
    pub sapiens_task: Option<SapiensTask>,
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
            trt_int8_enable: false,
            trt_fp16_enable: false,
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
            use_low_res_mask: None,
            sapiens_task: None,
            task: Task::Untitled,
        }
    }
}

impl Options {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_task(mut self, task: Task) -> Self {
        self.task = task;
        self
    }

    pub fn with_model(mut self, onnx_path: &str) -> Result<Self> {
        self.onnx_path = Hub::new()?.fetch(onnx_path)?.commit()?;
        Ok(self)
    }

    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    pub fn with_batch(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    pub fn with_dry_run(mut self, n: usize) -> Self {
        self.num_dry_run = n;
        self
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
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

    pub fn with_trt_fp16(mut self, x: bool) -> Self {
        self.trt_fp16_enable = x;
        self
    }

    pub fn with_yolo_task(mut self, x: YOLOTask) -> Self {
        self.yolo_task = Some(x);
        self
    }

    pub fn with_sapiens_task(mut self, x: SapiensTask) -> Self {
        self.sapiens_task = Some(x);
        self
    }

    pub fn with_yolo_version(mut self, x: YOLOVersion) -> Self {
        self.yolo_version = Some(x);
        self
    }

    pub fn with_profile(mut self, profile: bool) -> Self {
        self.profile = profile;
        self
    }

    pub fn with_find_contours(mut self, x: bool) -> Self {
        self.find_contours = x;
        self
    }

    pub fn with_sam_kind(mut self, x: SamKind) -> Self {
        self.sam_kind = Some(x);
        self
    }

    pub fn use_low_res_mask(mut self, x: bool) -> Self {
        self.use_low_res_mask = Some(x);
        self
    }

    pub fn with_names(mut self, names: &[&str]) -> Self {
        self.names = Some(names.iter().map(|x| x.to_string()).collect::<Vec<String>>());
        self
    }

    pub fn with_names2(mut self, names: &[&str]) -> Self {
        self.names2 = Some(names.iter().map(|x| x.to_string()).collect::<Vec<String>>());
        self
    }

    pub fn with_names3(mut self, names: &[&str]) -> Self {
        self.names3 = Some(names.iter().map(|x| x.to_string()).collect::<Vec<String>>());
        self
    }

    pub fn with_vocab(mut self, vocab: &str) -> Result<Self> {
        self.vocab = Some(Hub::new()?.fetch(vocab)?.commit()?);
        Ok(self)
    }

    pub fn with_context_length(mut self, n: usize) -> Self {
        self.context_length = Some(n);
        self
    }

    pub fn with_tokenizer(mut self, tokenizer: &str) -> Result<Self> {
        self.tokenizer = Some(Hub::new()?.fetch(tokenizer)?.commit()?);
        Ok(self)
    }

    pub fn with_unclip_ratio(mut self, x: f32) -> Self {
        self.unclip_ratio = x;
        self
    }

    pub fn with_min_width(mut self, x: f32) -> Self {
        self.min_width = Some(x);
        self
    }

    pub fn with_min_height(mut self, x: f32) -> Self {
        self.min_height = Some(x);
        self
    }

    pub fn with_yolo_preds(mut self, x: YOLOPreds) -> Self {
        self.yolo_preds = Some(x);
        self
    }

    pub fn with_nc(mut self, nc: usize) -> Self {
        self.nc = Some(nc);
        self
    }

    pub fn with_nk(mut self, nk: usize) -> Self {
        self.nk = Some(nk);
        self
    }

    pub fn with_iou(mut self, x: f32) -> Self {
        self.iou = Some(x);
        self
    }

    pub fn with_confs(mut self, x: &[f32]) -> Self {
        self.confs = x.to_vec();
        self
    }

    pub fn with_confs2(mut self, x: &[f32]) -> Self {
        self.confs2 = x.to_vec();
        self
    }

    pub fn with_confs3(mut self, x: &[f32]) -> Self {
        self.confs3 = x.to_vec();
        self
    }

    pub fn with_kconfs(mut self, kconfs: &[f32]) -> Self {
        self.kconfs = kconfs.to_vec();
        self
    }

    pub fn with_ixx(mut self, i: usize, ii: usize, x: MinOptMax) -> Self {
        self.iiixs.push(Iiix::from((i, ii, x)));
        self
    }
}
