use crate::{auto_load, Device, MinOptMax};

#[derive(Debug, Clone)]
pub struct Options {
    pub onnx_path: String,
    pub device: Device,
    pub profile: bool,
    pub num_dry_run: usize,
    pub i00: Option<MinOptMax>, // 1st input, axis 0, batch usually
    pub i01: Option<MinOptMax>, // 1st input, axis 1
    pub i02: Option<MinOptMax>,
    pub i03: Option<MinOptMax>,
    pub i04: Option<MinOptMax>,
    pub i05: Option<MinOptMax>,
    pub i10: Option<MinOptMax>, // 2nd input, axis 0
    pub i11: Option<MinOptMax>, // 2nd input, axis 1
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

    // trt related
    pub trt_engine_cache_enable: bool,
    pub trt_int8_enable: bool,
    pub trt_fp16_enable: bool,

    // options for Vision and Language models
    pub nc: Option<usize>,
    pub nk: Option<usize>,
    pub nm: Option<usize>,
    pub confs: Vec<f32>,
    pub kconfs: Vec<f32>,
    pub iou: f32,
    pub apply_nms: bool,
    pub tokenizer: Option<String>,
    pub vocab: Option<String>,
    pub names: Option<Vec<String>>, // class names
    pub anchors_first: bool,        // otuput format: [bs, anchors/na, pos+nc+nm]
    pub min_width: Option<f32>,
    pub min_height: Option<f32>,
    pub unclip_ratio: f32, // DB
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
            trt_engine_cache_enable: true,
            trt_int8_enable: false,
            trt_fp16_enable: false,
            nc: None,
            nk: None,
            nm: None,
            confs: vec![0.4f32],
            kconfs: vec![0.5f32],
            iou: 0.45f32,
            apply_nms: true,
            tokenizer: None,
            vocab: None,
            names: None,
            anchors_first: false,
            min_width: None,
            min_height: None,
            unclip_ratio: 1.5,
        }
    }
}

impl Options {
    pub fn with_model(mut self, onnx_path: &str) -> Self {
        self.onnx_path = auto_load(onnx_path).unwrap();
        self
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

    pub fn with_profile(mut self, profile: bool) -> Self {
        self.profile = profile;
        self
    }

    pub fn with_names(mut self, names: &[&str]) -> Self {
        self.names = Some(names.iter().map(|x| x.to_string()).collect::<Vec<String>>());
        self
    }

    pub fn with_vocab(mut self, vocab: &str) -> Self {
        self.vocab = Some(auto_load(vocab).unwrap());
        self
    }

    pub fn with_tokenizer(mut self, tokenizer: &str) -> Self {
        self.tokenizer = Some(auto_load(tokenizer).unwrap());
        self
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

    pub fn with_anchors_first(mut self) -> Self {
        self.anchors_first = true;
        self
    }

    pub fn with_nms(mut self, apply_nms: bool) -> Self {
        self.apply_nms = apply_nms;
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
        self.iou = x;
        self
    }

    pub fn with_confs(mut self, confs: &[f32]) -> Self {
        self.confs = confs.to_vec();
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
}
