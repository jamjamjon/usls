//! Options for build models.

use anyhow::Result;

use crate::{
    models::{SamKind, SapiensTask, YOLOPreds, YOLOTask, YOLOVersion},
    Device, Hub, MinOptMax, Task,
};

/// Options for building models
#[derive(Debug, Clone)]
pub struct Options {
    pub onnx_path: String,
    pub task: Task,
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
    pub i80: Option<MinOptMax>,
    pub i81: Option<MinOptMax>,
    pub i82: Option<MinOptMax>,
    pub i83: Option<MinOptMax>,
    pub i84: Option<MinOptMax>,
    pub i85: Option<MinOptMax>,
    pub i90: Option<MinOptMax>,
    pub i91: Option<MinOptMax>,
    pub i92: Option<MinOptMax>,
    pub i93: Option<MinOptMax>,
    pub i94: Option<MinOptMax>,
    pub i95: Option<MinOptMax>,
    pub i100: Option<MinOptMax>,
    pub i101: Option<MinOptMax>,
    pub i102: Option<MinOptMax>,
    pub i103: Option<MinOptMax>,
    pub i104: Option<MinOptMax>,
    pub i105: Option<MinOptMax>,
    pub i110: Option<MinOptMax>,
    pub i111: Option<MinOptMax>,
    pub i112: Option<MinOptMax>,
    pub i113: Option<MinOptMax>,
    pub i114: Option<MinOptMax>,
    pub i115: Option<MinOptMax>,
    pub i120: Option<MinOptMax>,
    pub i121: Option<MinOptMax>,
    pub i122: Option<MinOptMax>,
    pub i123: Option<MinOptMax>,
    pub i124: Option<MinOptMax>,
    pub i125: Option<MinOptMax>,
    pub i130: Option<MinOptMax>,
    pub i131: Option<MinOptMax>,
    pub i132: Option<MinOptMax>,
    pub i133: Option<MinOptMax>,
    pub i134: Option<MinOptMax>,
    pub i135: Option<MinOptMax>,
    pub i140: Option<MinOptMax>,
    pub i141: Option<MinOptMax>,
    pub i142: Option<MinOptMax>,
    pub i143: Option<MinOptMax>,
    pub i144: Option<MinOptMax>,
    pub i145: Option<MinOptMax>,
    pub i150: Option<MinOptMax>,
    pub i151: Option<MinOptMax>,
    pub i152: Option<MinOptMax>,
    pub i153: Option<MinOptMax>,
    pub i154: Option<MinOptMax>,
    pub i155: Option<MinOptMax>,
    pub i160: Option<MinOptMax>,
    pub i161: Option<MinOptMax>,
    pub i162: Option<MinOptMax>,
    pub i163: Option<MinOptMax>,
    pub i164: Option<MinOptMax>,
    pub i165: Option<MinOptMax>,
    pub i170: Option<MinOptMax>,
    pub i171: Option<MinOptMax>,
    pub i172: Option<MinOptMax>,
    pub i173: Option<MinOptMax>,
    pub i174: Option<MinOptMax>,
    pub i175: Option<MinOptMax>,
    pub i180: Option<MinOptMax>,
    pub i181: Option<MinOptMax>,
    pub i182: Option<MinOptMax>,
    pub i183: Option<MinOptMax>,
    pub i184: Option<MinOptMax>,
    pub i185: Option<MinOptMax>,
    pub i190: Option<MinOptMax>,
    pub i191: Option<MinOptMax>,
    pub i192: Option<MinOptMax>,
    pub i193: Option<MinOptMax>,
    pub i194: Option<MinOptMax>,
    pub i195: Option<MinOptMax>,
    pub i200: Option<MinOptMax>,
    pub i201: Option<MinOptMax>,
    pub i202: Option<MinOptMax>,
    pub i203: Option<MinOptMax>,
    pub i204: Option<MinOptMax>,
    pub i205: Option<MinOptMax>,
    pub i210: Option<MinOptMax>,
    pub i211: Option<MinOptMax>,
    pub i212: Option<MinOptMax>,
    pub i213: Option<MinOptMax>,
    pub i214: Option<MinOptMax>,
    pub i215: Option<MinOptMax>,
    pub i220: Option<MinOptMax>,
    pub i221: Option<MinOptMax>,
    pub i222: Option<MinOptMax>,
    pub i223: Option<MinOptMax>,
    pub i224: Option<MinOptMax>,
    pub i225: Option<MinOptMax>,
    pub i230: Option<MinOptMax>,
    pub i231: Option<MinOptMax>,
    pub i232: Option<MinOptMax>,
    pub i233: Option<MinOptMax>,
    pub i234: Option<MinOptMax>,
    pub i235: Option<MinOptMax>,
    pub i240: Option<MinOptMax>,
    pub i241: Option<MinOptMax>,
    pub i242: Option<MinOptMax>,
    pub i243: Option<MinOptMax>,
    pub i244: Option<MinOptMax>,
    pub i245: Option<MinOptMax>,
    pub i250: Option<MinOptMax>,
    pub i251: Option<MinOptMax>,
    pub i252: Option<MinOptMax>,
    pub i253: Option<MinOptMax>,
    pub i254: Option<MinOptMax>,
    pub i255: Option<MinOptMax>,
    pub i260: Option<MinOptMax>,
    pub i261: Option<MinOptMax>,
    pub i262: Option<MinOptMax>,
    pub i263: Option<MinOptMax>,
    pub i264: Option<MinOptMax>,
    pub i265: Option<MinOptMax>,
    pub i270: Option<MinOptMax>,
    pub i271: Option<MinOptMax>,
    pub i272: Option<MinOptMax>,
    pub i273: Option<MinOptMax>,
    pub i274: Option<MinOptMax>,
    pub i275: Option<MinOptMax>,
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
            num_dry_run: 3,
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
            i80: None,
            i81: None,
            i82: None,
            i83: None,
            i84: None,
            i85: None,
            i90: None,
            i91: None,
            i92: None,
            i93: None,
            i94: None,
            i95: None,
            i100: None,
            i101: None,
            i102: None,
            i103: None,
            i104: None,
            i105: None,
            i110: None,
            i111: None,
            i112: None,
            i113: None,
            i114: None,
            i115: None,
            i120: None,
            i121: None,
            i122: None,
            i123: None,
            i124: None,
            i125: None,
            i130: None,
            i131: None,
            i132: None,
            i133: None,
            i134: None,
            i135: None,
            i140: None,
            i141: None,
            i142: None,
            i143: None,
            i144: None,
            i145: None,
            i150: None,
            i151: None,
            i152: None,
            i153: None,
            i154: None,
            i155: None,
            i160: None,
            i161: None,
            i162: None,
            i163: None,
            i164: None,
            i165: None,
            i170: None,
            i171: None,
            i172: None,
            i173: None,
            i174: None,
            i175: None,
            i180: None,
            i181: None,
            i182: None,
            i183: None,
            i184: None,
            i185: None,
            i190: None,
            i191: None,
            i192: None,
            i193: None,
            i194: None,
            i195: None,
            i200: None,
            i201: None,
            i202: None,
            i203: None,
            i204: None,
            i205: None,
            i210: None,
            i211: None,
            i212: None,
            i213: None,
            i214: None,
            i215: None,
            i220: None,
            i221: None,
            i222: None,
            i223: None,
            i224: None,
            i225: None,
            i230: None,
            i231: None,
            i232: None,
            i233: None,
            i234: None,
            i235: None,
            i240: None,
            i241: None,
            i242: None,
            i243: None,
            i244: None,
            i245: None,
            i250: None,
            i251: None,
            i252: None,
            i253: None,
            i254: None,
            i255: None,
            i260: None,
            i261: None,
            i262: None,
            i263: None,
            i264: None,
            i265: None,
            i270: None,
            i271: None,
            i272: None,
            i273: None,
            i274: None,
            i275: None,
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

    pub fn with_i80(mut self, x: MinOptMax) -> Self {
        self.i80 = Some(x);
        self
    }

    pub fn with_i81(mut self, x: MinOptMax) -> Self {
        self.i81 = Some(x);
        self
    }

    pub fn with_i82(mut self, x: MinOptMax) -> Self {
        self.i82 = Some(x);
        self
    }

    pub fn with_i83(mut self, x: MinOptMax) -> Self {
        self.i83 = Some(x);
        self
    }

    pub fn with_i84(mut self, x: MinOptMax) -> Self {
        self.i84 = Some(x);
        self
    }

    pub fn with_i85(mut self, x: MinOptMax) -> Self {
        self.i85 = Some(x);
        self
    }

    pub fn with_i90(mut self, x: MinOptMax) -> Self {
        self.i90 = Some(x);
        self
    }

    pub fn with_i91(mut self, x: MinOptMax) -> Self {
        self.i91 = Some(x);
        self
    }

    pub fn with_i92(mut self, x: MinOptMax) -> Self {
        self.i92 = Some(x);
        self
    }

    pub fn with_i93(mut self, x: MinOptMax) -> Self {
        self.i93 = Some(x);
        self
    }

    pub fn with_i94(mut self, x: MinOptMax) -> Self {
        self.i94 = Some(x);
        self
    }

    pub fn with_i95(mut self, x: MinOptMax) -> Self {
        self.i95 = Some(x);
        self
    }

    pub fn with_i100(mut self, x: MinOptMax) -> Self {
        self.i100 = Some(x);
        self
    }

    pub fn with_i101(mut self, x: MinOptMax) -> Self {
        self.i101 = Some(x);
        self
    }

    pub fn with_i102(mut self, x: MinOptMax) -> Self {
        self.i102 = Some(x);
        self
    }

    pub fn with_i103(mut self, x: MinOptMax) -> Self {
        self.i103 = Some(x);
        self
    }

    pub fn with_i104(mut self, x: MinOptMax) -> Self {
        self.i104 = Some(x);
        self
    }

    pub fn with_i105(mut self, x: MinOptMax) -> Self {
        self.i105 = Some(x);
        self
    }

    pub fn with_i110(mut self, x: MinOptMax) -> Self {
        self.i110 = Some(x);
        self
    }

    pub fn with_i111(mut self, x: MinOptMax) -> Self {
        self.i111 = Some(x);
        self
    }

    pub fn with_i112(mut self, x: MinOptMax) -> Self {
        self.i112 = Some(x);
        self
    }

    pub fn with_i113(mut self, x: MinOptMax) -> Self {
        self.i113 = Some(x);
        self
    }

    pub fn with_i114(mut self, x: MinOptMax) -> Self {
        self.i114 = Some(x);
        self
    }

    pub fn with_i115(mut self, x: MinOptMax) -> Self {
        self.i115 = Some(x);
        self
    }

    pub fn with_i120(mut self, x: MinOptMax) -> Self {
        self.i120 = Some(x);
        self
    }

    pub fn with_i121(mut self, x: MinOptMax) -> Self {
        self.i121 = Some(x);
        self
    }

    pub fn with_i122(mut self, x: MinOptMax) -> Self {
        self.i122 = Some(x);
        self
    }

    pub fn with_i123(mut self, x: MinOptMax) -> Self {
        self.i123 = Some(x);
        self
    }

    pub fn with_i124(mut self, x: MinOptMax) -> Self {
        self.i124 = Some(x);
        self
    }

    pub fn with_i125(mut self, x: MinOptMax) -> Self {
        self.i125 = Some(x);
        self
    }

    pub fn with_i130(mut self, x: MinOptMax) -> Self {
        self.i130 = Some(x);
        self
    }

    pub fn with_i131(mut self, x: MinOptMax) -> Self {
        self.i131 = Some(x);
        self
    }

    pub fn with_i132(mut self, x: MinOptMax) -> Self {
        self.i132 = Some(x);
        self
    }

    pub fn with_i133(mut self, x: MinOptMax) -> Self {
        self.i133 = Some(x);
        self
    }

    pub fn with_i134(mut self, x: MinOptMax) -> Self {
        self.i134 = Some(x);
        self
    }

    pub fn with_i135(mut self, x: MinOptMax) -> Self {
        self.i135 = Some(x);
        self
    }

    pub fn with_i140(mut self, x: MinOptMax) -> Self {
        self.i140 = Some(x);
        self
    }

    pub fn with_i141(mut self, x: MinOptMax) -> Self {
        self.i141 = Some(x);
        self
    }

    pub fn with_i142(mut self, x: MinOptMax) -> Self {
        self.i142 = Some(x);
        self
    }

    pub fn with_i143(mut self, x: MinOptMax) -> Self {
        self.i143 = Some(x);
        self
    }

    pub fn with_i144(mut self, x: MinOptMax) -> Self {
        self.i144 = Some(x);
        self
    }

    pub fn with_i145(mut self, x: MinOptMax) -> Self {
        self.i145 = Some(x);
        self
    }

    pub fn with_i150(mut self, x: MinOptMax) -> Self {
        self.i150 = Some(x);
        self
    }

    pub fn with_i151(mut self, x: MinOptMax) -> Self {
        self.i151 = Some(x);
        self
    }

    pub fn with_i152(mut self, x: MinOptMax) -> Self {
        self.i152 = Some(x);
        self
    }

    pub fn with_i153(mut self, x: MinOptMax) -> Self {
        self.i153 = Some(x);
        self
    }

    pub fn with_i154(mut self, x: MinOptMax) -> Self {
        self.i154 = Some(x);
        self
    }

    pub fn with_i155(mut self, x: MinOptMax) -> Self {
        self.i155 = Some(x);
        self
    }

    pub fn with_i160(mut self, x: MinOptMax) -> Self {
        self.i160 = Some(x);
        self
    }

    pub fn with_i161(mut self, x: MinOptMax) -> Self {
        self.i161 = Some(x);
        self
    }

    pub fn with_i162(mut self, x: MinOptMax) -> Self {
        self.i162 = Some(x);
        self
    }

    pub fn with_i163(mut self, x: MinOptMax) -> Self {
        self.i163 = Some(x);
        self
    }

    pub fn with_i164(mut self, x: MinOptMax) -> Self {
        self.i164 = Some(x);
        self
    }

    pub fn with_i165(mut self, x: MinOptMax) -> Self {
        self.i165 = Some(x);
        self
    }

    pub fn with_i170(mut self, x: MinOptMax) -> Self {
        self.i170 = Some(x);
        self
    }

    pub fn with_i171(mut self, x: MinOptMax) -> Self {
        self.i171 = Some(x);
        self
    }

    pub fn with_i172(mut self, x: MinOptMax) -> Self {
        self.i172 = Some(x);
        self
    }

    pub fn with_i173(mut self, x: MinOptMax) -> Self {
        self.i173 = Some(x);
        self
    }

    pub fn with_i174(mut self, x: MinOptMax) -> Self {
        self.i174 = Some(x);
        self
    }

    pub fn with_i175(mut self, x: MinOptMax) -> Self {
        self.i175 = Some(x);
        self
    }

    pub fn with_i180(mut self, x: MinOptMax) -> Self {
        self.i180 = Some(x);
        self
    }

    pub fn with_i181(mut self, x: MinOptMax) -> Self {
        self.i181 = Some(x);
        self
    }

    pub fn with_i182(mut self, x: MinOptMax) -> Self {
        self.i182 = Some(x);
        self
    }

    pub fn with_i183(mut self, x: MinOptMax) -> Self {
        self.i183 = Some(x);
        self
    }

    pub fn with_i184(mut self, x: MinOptMax) -> Self {
        self.i184 = Some(x);
        self
    }

    pub fn with_i185(mut self, x: MinOptMax) -> Self {
        self.i185 = Some(x);
        self
    }

    pub fn with_i190(mut self, x: MinOptMax) -> Self {
        self.i190 = Some(x);
        self
    }

    pub fn with_i191(mut self, x: MinOptMax) -> Self {
        self.i191 = Some(x);
        self
    }

    pub fn with_i192(mut self, x: MinOptMax) -> Self {
        self.i192 = Some(x);
        self
    }

    pub fn with_i193(mut self, x: MinOptMax) -> Self {
        self.i193 = Some(x);
        self
    }

    pub fn with_i194(mut self, x: MinOptMax) -> Self {
        self.i194 = Some(x);
        self
    }

    pub fn with_i195(mut self, x: MinOptMax) -> Self {
        self.i195 = Some(x);
        self
    }

    pub fn with_i200(mut self, x: MinOptMax) -> Self {
        self.i200 = Some(x);
        self
    }

    pub fn with_i201(mut self, x: MinOptMax) -> Self {
        self.i201 = Some(x);
        self
    }

    pub fn with_i202(mut self, x: MinOptMax) -> Self {
        self.i202 = Some(x);
        self
    }

    pub fn with_i203(mut self, x: MinOptMax) -> Self {
        self.i203 = Some(x);
        self
    }

    pub fn with_i204(mut self, x: MinOptMax) -> Self {
        self.i204 = Some(x);
        self
    }

    pub fn with_i205(mut self, x: MinOptMax) -> Self {
        self.i205 = Some(x);
        self
    }

    pub fn with_i210(mut self, x: MinOptMax) -> Self {
        self.i210 = Some(x);
        self
    }

    pub fn with_i211(mut self, x: MinOptMax) -> Self {
        self.i211 = Some(x);
        self
    }

    pub fn with_i212(mut self, x: MinOptMax) -> Self {
        self.i212 = Some(x);
        self
    }

    pub fn with_i213(mut self, x: MinOptMax) -> Self {
        self.i213 = Some(x);
        self
    }

    pub fn with_i214(mut self, x: MinOptMax) -> Self {
        self.i214 = Some(x);
        self
    }

    pub fn with_i215(mut self, x: MinOptMax) -> Self {
        self.i215 = Some(x);
        self
    }

    pub fn with_i220(mut self, x: MinOptMax) -> Self {
        self.i220 = Some(x);
        self
    }

    pub fn with_i221(mut self, x: MinOptMax) -> Self {
        self.i221 = Some(x);
        self
    }

    pub fn with_i222(mut self, x: MinOptMax) -> Self {
        self.i222 = Some(x);
        self
    }

    pub fn with_i223(mut self, x: MinOptMax) -> Self {
        self.i223 = Some(x);
        self
    }

    pub fn with_i224(mut self, x: MinOptMax) -> Self {
        self.i224 = Some(x);
        self
    }

    pub fn with_i225(mut self, x: MinOptMax) -> Self {
        self.i225 = Some(x);
        self
    }

    pub fn with_i230(mut self, x: MinOptMax) -> Self {
        self.i230 = Some(x);
        self
    }

    pub fn with_i231(mut self, x: MinOptMax) -> Self {
        self.i231 = Some(x);
        self
    }

    pub fn with_i232(mut self, x: MinOptMax) -> Self {
        self.i232 = Some(x);
        self
    }

    pub fn with_i233(mut self, x: MinOptMax) -> Self {
        self.i233 = Some(x);
        self
    }

    pub fn with_i234(mut self, x: MinOptMax) -> Self {
        self.i234 = Some(x);
        self
    }

    pub fn with_i235(mut self, x: MinOptMax) -> Self {
        self.i235 = Some(x);
        self
    }

    pub fn with_i240(mut self, x: MinOptMax) -> Self {
        self.i240 = Some(x);
        self
    }

    pub fn with_i241(mut self, x: MinOptMax) -> Self {
        self.i241 = Some(x);
        self
    }

    pub fn with_i242(mut self, x: MinOptMax) -> Self {
        self.i242 = Some(x);
        self
    }

    pub fn with_i243(mut self, x: MinOptMax) -> Self {
        self.i243 = Some(x);
        self
    }

    pub fn with_i244(mut self, x: MinOptMax) -> Self {
        self.i244 = Some(x);
        self
    }

    pub fn with_i245(mut self, x: MinOptMax) -> Self {
        self.i245 = Some(x);
        self
    }

    pub fn with_i250(mut self, x: MinOptMax) -> Self {
        self.i250 = Some(x);
        self
    }

    pub fn with_i251(mut self, x: MinOptMax) -> Self {
        self.i251 = Some(x);
        self
    }

    pub fn with_i252(mut self, x: MinOptMax) -> Self {
        self.i252 = Some(x);
        self
    }

    pub fn with_i253(mut self, x: MinOptMax) -> Self {
        self.i253 = Some(x);
        self
    }

    pub fn with_i254(mut self, x: MinOptMax) -> Self {
        self.i254 = Some(x);
        self
    }

    pub fn with_i255(mut self, x: MinOptMax) -> Self {
        self.i255 = Some(x);
        self
    }
    pub fn with_i260(mut self, x: MinOptMax) -> Self {
        self.i260 = Some(x);
        self
    }

    pub fn with_i261(mut self, x: MinOptMax) -> Self {
        self.i261 = Some(x);
        self
    }

    pub fn with_i262(mut self, x: MinOptMax) -> Self {
        self.i262 = Some(x);
        self
    }

    pub fn with_i263(mut self, x: MinOptMax) -> Self {
        self.i263 = Some(x);
        self
    }

    pub fn with_i264(mut self, x: MinOptMax) -> Self {
        self.i264 = Some(x);
        self
    }

    pub fn with_i265(mut self, x: MinOptMax) -> Self {
        self.i265 = Some(x);
        self
    }

    pub fn with_i270(mut self, x: MinOptMax) -> Self {
        self.i270 = Some(x);
        self
    }

    pub fn with_i271(mut self, x: MinOptMax) -> Self {
        self.i271 = Some(x);
        self
    }

    pub fn with_i272(mut self, x: MinOptMax) -> Self {
        self.i272 = Some(x);
        self
    }

    pub fn with_i273(mut self, x: MinOptMax) -> Self {
        self.i273 = Some(x);
        self
    }

    pub fn with_i274(mut self, x: MinOptMax) -> Self {
        self.i274 = Some(x);
        self
    }

    pub fn with_i275(mut self, x: MinOptMax) -> Self {
        self.i275 = Some(x);
        self
    }
}
