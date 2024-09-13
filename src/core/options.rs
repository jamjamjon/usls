//! Options for build models.

use anyhow::Result;

use crate::{
    models::{SamKind, SapiensTask, YOLOPreds, YOLOTask, YOLOVersion},
    Device, Hub, MinOptMax,
};

/// Options for building models
#[derive(Debug, Clone)]
pub struct Options {
    pub onnx_path: String,
    pub device: Device,
    pub profile: bool,
    pub num_dry_run: usize,
    pub i00: Option<MinOptMax>, // the 1st input, axis 0, batch usually
    pub i01: Option<MinOptMax>, // the 1st input, axis 1
    pub i02: Option<MinOptMax>,
    pub i03: Option<MinOptMax>,
    pub i04: Option<MinOptMax>,
    pub i05: Option<MinOptMax>,
    pub i10: Option<MinOptMax>, // the 2nd input, axis 0
    pub i11: Option<MinOptMax>, // the 2nd input, axis 1
    pub i12: Option<MinOptMax>,
    pub i13: Option<MinOptMax>,
    pub i14: Option<MinOptMax>,
    pub i15: Option<MinOptMax>,
    pub i20: Option<MinOptMax>,
    pub i21: Option<MinOptMax>,
    pub i22: Option<MinOptMax>,
    pub i23: Option<MinOptMax>,
    pub i24: Option<MinOptMax>,
    pub i25: Option<MinOptMax>,
    pub i30: Option<MinOptMax>,
    pub i31: Option<MinOptMax>,
    pub i32_: Option<MinOptMax>,
    pub i33: Option<MinOptMax>,
    pub i34: Option<MinOptMax>,
    pub i35: Option<MinOptMax>,
    pub i40: Option<MinOptMax>,
    pub i41: Option<MinOptMax>,
    pub i42: Option<MinOptMax>,
    pub i43: Option<MinOptMax>,
    pub i44: Option<MinOptMax>,
    pub i45: Option<MinOptMax>,
    pub i50: Option<MinOptMax>,
    pub i51: Option<MinOptMax>,
    pub i52: Option<MinOptMax>,
    pub i53: Option<MinOptMax>,
    pub i54: Option<MinOptMax>,
    pub i55: Option<MinOptMax>,
    pub i60: Option<MinOptMax>,
    pub i61: Option<MinOptMax>,
    pub i62: Option<MinOptMax>,
    pub i63: Option<MinOptMax>,
    pub i64_: Option<MinOptMax>,
    pub i65: Option<MinOptMax>,
    pub i70: Option<MinOptMax>,
    pub i71: Option<MinOptMax>,
    pub i72: Option<MinOptMax>,
    pub i73: Option<MinOptMax>,
    pub i74: Option<MinOptMax>,
    pub i75: Option<MinOptMax>,
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
            num_dry_run: 5,
            i00: None,
            i01: None,
            i02: None,
            i03: None,
            i04: None,
            i05: None,
            i10: None,
            i11: None,
            i12: None,
            i13: None,
            i14: None,
            i15: None,
            i20: None,
            i21: None,
            i22: None,
            i23: None,
            i24: None,
            i25: None,
            i30: None,
            i31: None,
            i32_: None,
            i33: None,
            i34: None,
            i35: None,
            i40: None,
            i41: None,
            i42: None,
            i43: None,
            i44: None,
            i45: None,
            i50: None,
            i51: None,
            i52: None,
            i53: None,
            i54: None,
            i55: None,
            i60: None,
            i61: None,
            i62: None,
            i63: None,
            i64_: None,
            i65: None,
            i70: None,
            i71: None,
            i72: None,
            i73: None,
            i74: None,
            i75: None,
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

    pub fn with_dry_run(mut self, n: usize) -> Self {
        self.num_dry_run = n;
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

    pub fn with_fp16(mut self, x: bool) -> Self {
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

    pub fn with_i00(mut self, x: MinOptMax) -> Self {
        self.i00 = Some(x);
        self
    }

    pub fn with_i01(mut self, x: MinOptMax) -> Self {
        self.i01 = Some(x);
        self
    }

    pub fn with_i02(mut self, x: MinOptMax) -> Self {
        self.i02 = Some(x);
        self
    }

    pub fn with_i03(mut self, x: MinOptMax) -> Self {
        self.i03 = Some(x);
        self
    }

    pub fn with_i04(mut self, x: MinOptMax) -> Self {
        self.i04 = Some(x);
        self
    }

    pub fn with_i05(mut self, x: MinOptMax) -> Self {
        self.i05 = Some(x);
        self
    }

    pub fn with_i10(mut self, x: MinOptMax) -> Self {
        self.i10 = Some(x);
        self
    }

    pub fn with_i11(mut self, x: MinOptMax) -> Self {
        self.i11 = Some(x);
        self
    }

    pub fn with_i12(mut self, x: MinOptMax) -> Self {
        self.i12 = Some(x);
        self
    }

    pub fn with_i13(mut self, x: MinOptMax) -> Self {
        self.i13 = Some(x);
        self
    }

    pub fn with_i14(mut self, x: MinOptMax) -> Self {
        self.i14 = Some(x);
        self
    }

    pub fn with_i15(mut self, x: MinOptMax) -> Self {
        self.i15 = Some(x);
        self
    }

    pub fn with_i20(mut self, x: MinOptMax) -> Self {
        self.i20 = Some(x);
        self
    }

    pub fn with_i21(mut self, x: MinOptMax) -> Self {
        self.i21 = Some(x);
        self
    }

    pub fn with_i22(mut self, x: MinOptMax) -> Self {
        self.i22 = Some(x);
        self
    }

    pub fn with_i23(mut self, x: MinOptMax) -> Self {
        self.i23 = Some(x);
        self
    }

    pub fn with_i24(mut self, x: MinOptMax) -> Self {
        self.i24 = Some(x);
        self
    }

    pub fn with_i25(mut self, x: MinOptMax) -> Self {
        self.i25 = Some(x);
        self
    }

    pub fn with_i30(mut self, x: MinOptMax) -> Self {
        self.i30 = Some(x);
        self
    }

    pub fn with_i31(mut self, x: MinOptMax) -> Self {
        self.i31 = Some(x);
        self
    }

    pub fn with_i32_(mut self, x: MinOptMax) -> Self {
        self.i32_ = Some(x);
        self
    }

    pub fn with_i33(mut self, x: MinOptMax) -> Self {
        self.i33 = Some(x);
        self
    }

    pub fn with_i34(mut self, x: MinOptMax) -> Self {
        self.i34 = Some(x);
        self
    }

    pub fn with_i35(mut self, x: MinOptMax) -> Self {
        self.i35 = Some(x);
        self
    }

    pub fn with_i40(mut self, x: MinOptMax) -> Self {
        self.i40 = Some(x);
        self
    }

    pub fn with_i41(mut self, x: MinOptMax) -> Self {
        self.i41 = Some(x);
        self
    }

    pub fn with_i42(mut self, x: MinOptMax) -> Self {
        self.i42 = Some(x);
        self
    }

    pub fn with_i43(mut self, x: MinOptMax) -> Self {
        self.i43 = Some(x);
        self
    }

    pub fn with_i44(mut self, x: MinOptMax) -> Self {
        self.i44 = Some(x);
        self
    }

    pub fn with_i45(mut self, x: MinOptMax) -> Self {
        self.i45 = Some(x);
        self
    }

    pub fn with_i50(mut self, x: MinOptMax) -> Self {
        self.i50 = Some(x);
        self
    }

    pub fn with_i51(mut self, x: MinOptMax) -> Self {
        self.i51 = Some(x);
        self
    }

    pub fn with_i52(mut self, x: MinOptMax) -> Self {
        self.i52 = Some(x);
        self
    }

    pub fn with_i53(mut self, x: MinOptMax) -> Self {
        self.i53 = Some(x);
        self
    }

    pub fn with_i54(mut self, x: MinOptMax) -> Self {
        self.i54 = Some(x);
        self
    }

    pub fn with_i55(mut self, x: MinOptMax) -> Self {
        self.i55 = Some(x);
        self
    }

    pub fn with_i60(mut self, x: MinOptMax) -> Self {
        self.i60 = Some(x);
        self
    }

    pub fn with_i61(mut self, x: MinOptMax) -> Self {
        self.i61 = Some(x);
        self
    }

    pub fn with_i62(mut self, x: MinOptMax) -> Self {
        self.i62 = Some(x);
        self
    }

    pub fn with_i63(mut self, x: MinOptMax) -> Self {
        self.i63 = Some(x);
        self
    }

    pub fn with_i64(mut self, x: MinOptMax) -> Self {
        self.i64_ = Some(x);
        self
    }

    pub fn with_i65(mut self, x: MinOptMax) -> Self {
        self.i65 = Some(x);
        self
    }

    pub fn with_i70(mut self, x: MinOptMax) -> Self {
        self.i70 = Some(x);
        self
    }

    pub fn with_i71(mut self, x: MinOptMax) -> Self {
        self.i71 = Some(x);
        self
    }

    pub fn with_i72(mut self, x: MinOptMax) -> Self {
        self.i72 = Some(x);
        self
    }

    pub fn with_i73(mut self, x: MinOptMax) -> Self {
        self.i73 = Some(x);
        self
    }

    pub fn with_i74(mut self, x: MinOptMax) -> Self {
        self.i74 = Some(x);
        self
    }

    pub fn with_i75(mut self, x: MinOptMax) -> Self {
        self.i75 = Some(x);
        self
    }
}
