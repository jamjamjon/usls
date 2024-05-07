use anyhow::Result;
use half::f16;
use human_bytes::human_bytes;
use ndarray::{Array, IxDyn};
use ort::{
    ExecutionProvider, ExecutionProviderDispatch, Session, SessionBuilder, TensorElementType,
    TensorRTExecutionProvider, ValueType,
};
use prost::Message;
use std::collections::HashSet;

use crate::{
    home_dir, onnx, ops::make_divisible, Device, MinOptMax, Options, CHECK_MARK, CROSS_MARK,
    SAFE_CROSS_MARK,
};

/// ONNXRuntime Backend
#[derive(Debug)]
pub struct OrtEngine {
    session: Session,
    device: Device,
    inputs_minoptmax: Vec<Vec<MinOptMax>>,
    inames: Vec<String>,
    ishapes: Vec<Vec<isize>>,
    idtypes: Vec<TensorElementType>,
    onames: Vec<String>,
    oshapes: Vec<Vec<isize>>,
    odtypes: Vec<TensorElementType>,
    profile: bool,
    num_dry_run: usize,
    model_proto: onnx::ModelProto,
    params: usize,
    wbmems: usize,
}

impl OrtEngine {
    pub fn dry_run(&self) -> Result<()> {
        if self.num_dry_run == 0 {
            println!("{SAFE_CROSS_MARK} No dry run count specified, skipping the dry run.");
            return Ok(());
        }
        let mut xs: Vec<Array<f32, IxDyn>> = Vec::new();
        for i in self.inputs_minoptmax.iter() {
            let mut x: Vec<usize> = Vec::new();
            for i_ in i.iter() {
                x.push(i_.opt as usize);
            }
            let x: Array<f32, IxDyn> = Array::ones(x).into_dyn();
            xs.push(x);
        }
        for _ in 0..self.num_dry_run {
            self.run(xs.as_ref())?;
        }
        println!("{CHECK_MARK} Dry run x{}", self.num_dry_run);
        Ok(())
    }

    pub fn new(config: &Options) -> Result<Self> {
        // onnx graph
        let model_proto = Self::load_onnx(&config.onnx_path)?;
        let graph = match &model_proto.graph {
            Some(graph) => graph,
            None => anyhow::bail!("No graph found in this proto"),
        };

        // model params & mem
        let byte_alignment = 16; // 16 for SIMD; 8 for most
        let mut params: usize = 0;
        let mut wbmems: usize = 0;
        let mut initializer_names: HashSet<&str> = HashSet::new();
        for tensor_proto in graph.initializer.iter() {
            initializer_names.insert(&tensor_proto.name);
            let param = tensor_proto.dims.iter().product::<i64>() as usize;
            params += param;

            // mems
            let param = make_divisible(param, byte_alignment);
            let n = Self::nbytes_from_onnx_dtype_id(tensor_proto.data_type as usize);
            let wbmem = param * n;
            wbmems += wbmem;
        }

        // inputs
        let (ishapes, idtypes, inames) =
            Self::parse_onnx_value_info(&initializer_names, &graph.input);

        // outputs
        let (oshapes, odtypes, onames) =
            Self::parse_onnx_value_info(&initializer_names, &graph.output);

        // inputs minoptmax
        let mut inputs_minoptmax: Vec<Vec<MinOptMax>> = Vec::new();
        for (i, dims) in ishapes.iter().enumerate() {
            let mut v_: Vec<MinOptMax> = Vec::new();
            for (ii, &x) in dims.iter().enumerate() {
                let x_default: MinOptMax = (ishapes[i][ii], ishapes[i][ii], ishapes[i][ii]).into();
                let x: MinOptMax = match (i, ii) {
                    (0, 0) => Self::_set_ixx(x, &config.i00, i, ii).unwrap_or(x_default),
                    (0, 1) => Self::_set_ixx(x, &config.i01, i, ii).unwrap_or(x_default),
                    (0, 2) => Self::_set_ixx(x, &config.i02, i, ii).unwrap_or(x_default),
                    (0, 3) => Self::_set_ixx(x, &config.i03, i, ii).unwrap_or(x_default),
                    (0, 4) => Self::_set_ixx(x, &config.i04, i, ii).unwrap_or(x_default),
                    (0, 5) => Self::_set_ixx(x, &config.i05, i, ii).unwrap_or(x_default),
                    (1, 0) => Self::_set_ixx(x, &config.i10, i, ii).unwrap_or(x_default),
                    (1, 1) => Self::_set_ixx(x, &config.i11, i, ii).unwrap_or(x_default),
                    (1, 2) => Self::_set_ixx(x, &config.i12, i, ii).unwrap_or(x_default),
                    (1, 3) => Self::_set_ixx(x, &config.i13, i, ii).unwrap_or(x_default),
                    (1, 4) => Self::_set_ixx(x, &config.i14, i, ii).unwrap_or(x_default),
                    (1, 5) => Self::_set_ixx(x, &config.i15, i, ii).unwrap_or(x_default),
                    (2, 0) => Self::_set_ixx(x, &config.i20, i, ii).unwrap_or(x_default),
                    (2, 1) => Self::_set_ixx(x, &config.i21, i, ii).unwrap_or(x_default),
                    (2, 2) => Self::_set_ixx(x, &config.i22, i, ii).unwrap_or(x_default),
                    (2, 3) => Self::_set_ixx(x, &config.i23, i, ii).unwrap_or(x_default),
                    (2, 4) => Self::_set_ixx(x, &config.i24, i, ii).unwrap_or(x_default),
                    (2, 5) => Self::_set_ixx(x, &config.i25, i, ii).unwrap_or(x_default),
                    (3, 0) => Self::_set_ixx(x, &config.i30, i, ii).unwrap_or(x_default),
                    (3, 1) => Self::_set_ixx(x, &config.i31, i, ii).unwrap_or(x_default),
                    (3, 2) => Self::_set_ixx(x, &config.i32_, i, ii).unwrap_or(x_default),
                    (3, 3) => Self::_set_ixx(x, &config.i33, i, ii).unwrap_or(x_default),
                    (3, 4) => Self::_set_ixx(x, &config.i34, i, ii).unwrap_or(x_default),
                    (3, 5) => Self::_set_ixx(x, &config.i35, i, ii).unwrap_or(x_default),
                    _ => todo!(),
                };
                v_.push(x);
            }
            inputs_minoptmax.push(v_);
        }

        // build again
        ort::init().commit()?;
        let builder = Session::builder()?;
        let device = config.device.to_owned();
        let (_ep, s) = match device {
            Device::Trt(device_id) => Self::build_trt(
                &inames,
                &inputs_minoptmax,
                &builder,
                device_id,
                config.trt_int8_enable,
                config.trt_fp16_enable,
                config.trt_engine_cache_enable,
            )?,
            Device::Cuda(device_id) => Self::build_cuda(&builder, device_id)?,
            Device::CoreML(_) => {
                let coreml = ort::CoreMLExecutionProvider::default()
                    .with_subgraphs()
                    // .with_ane_only()
                    .build();
                if coreml.is_available()? && coreml.register(&builder).is_ok() {
                    // println!("{CHECK_MARK} Using CoreML");
                    (coreml, String::from("CoreML"))
                } else {
                    println!("{CROSS_MARK} CoreML initialization failed");
                    // println!("{CHECK_MARK} Using CPU");
                    (
                        ort::CPUExecutionProvider::default().build(),
                        String::from("CPU"),
                    )
                }
            }
            Device::Cpu(_) => {
                // println!("{CHECK_MARK} Using CPU");
                (
                    ort::CPUExecutionProvider::default().build(),
                    String::from("CPU"),
                )
            } // _ => todo!(),
        };
        let session = builder
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_model_from_file(&config.onnx_path)?;

        // summary
        // TODO: ort version, ort upgrade
        println!(
            "{CHECK_MARK} Using {s} | Dtype: {idtypes:?} | Opset: {} | Parameters: {} ({} {})",
            model_proto.opset_import[0].version,
            human_bytes(params as f64),
            sysinfo::System::long_os_version().unwrap_or_default(),
            sysinfo::System::cpu_arch().unwrap_or_default(),
        );

        Ok(Self {
            session,
            device,
            inputs_minoptmax,
            inames,
            ishapes,
            idtypes,
            onames,
            oshapes,
            odtypes,
            profile: config.profile,
            num_dry_run: config.num_dry_run,
            model_proto,
            params,
            wbmems,
        })
    }

    fn build_trt(
        inames: &[String],
        inputs_minoptmax: &[Vec<MinOptMax>],
        builder: &SessionBuilder,
        device_id: usize,
        int8_enable: bool,
        fp16_enable: bool,
        engine_cache_enable: bool,
    ) -> Result<(ExecutionProviderDispatch, String)> {
        // auto generate shapes
        let mut spec_min = String::new();
        let mut spec_opt = String::new();
        let mut spec_max = String::new();
        for (i, name) in inames.iter().enumerate() {
            if i != 0 {
                spec_min.push(',');
                spec_opt.push(',');
                spec_max.push(',');
            }
            let mut s_min = format!("{}:", name);
            let mut s_opt = format!("{}:", name);
            let mut s_max = format!("{}:", name);
            for d in inputs_minoptmax[i].iter() {
                let min_ = &format!("{}x", d.min);
                let opt_ = &format!("{}x", d.opt);
                let max_ = &format!("{}x", d.max);
                s_min += min_;
                s_opt += opt_;
                s_max += max_;
            }
            s_min.pop();
            s_opt.pop();
            s_max.pop();
            spec_min += &s_min;
            spec_opt += &s_opt;
            spec_max += &s_max;
        }
        let trt = TensorRTExecutionProvider::default()
            .with_device_id(device_id as i32)
            .with_int8(int8_enable)
            .with_fp16(fp16_enable)
            .with_engine_cache(engine_cache_enable)
            .with_engine_cache_path(format!(
                "{}/{}",
                home_dir(None).to_str().unwrap(),
                "trt-cache"
            ))
            .with_timing_cache(false)
            .with_profile_min_shapes(spec_min)
            .with_profile_opt_shapes(spec_opt)
            .with_profile_max_shapes(spec_max)
            .build();
        if trt.is_available()? && trt.register(builder).is_ok() {
            println!("\n🐢 Initial model serialization with TensorRT may require a wait...\n");
            Ok((trt, String::from("TensorRT")))
        } else {
            println!("{CROSS_MARK} TensorRT initialization failed. Try CUDA...");
            Self::build_cuda(builder, device_id)
        }
    }

    fn build_cuda(
        builder: &SessionBuilder,
        device_id: usize,
    ) -> Result<(ExecutionProviderDispatch, String)> {
        let cuda = ort::CUDAExecutionProvider::default()
            .with_device_id(device_id as i32)
            .build();
        if cuda.is_available()? && cuda.register(builder).is_ok() {
            // println!("{CHECK_MARK} Using CUDA");
            Ok((cuda, String::from("CUDA")))
        } else {
            println!("{CROSS_MARK} CUDA initialization failed");
            // println!("{CHECK_MARK} Using CPU");
            Ok((
                ort::CPUExecutionProvider::default().build(),
                String::from("CPU"),
            ))
        }
    }

    pub fn run(&self, xs: &[Array<f32, IxDyn>]) -> Result<Vec<Array<f32, IxDyn>>> {
        // input
        let mut xs_ = Vec::new();
        let t_pre = std::time::Instant::now();
        for (idtype, x) in self.idtypes.iter().zip(xs.iter()) {
            let x_ = match idtype {
                TensorElementType::Float32 => ort::Value::from_array(x.view())?,
                TensorElementType::Float16 => ort::Value::from_array(x.mapv(f16::from_f32).view())?,
                TensorElementType::Int32 => ort::Value::from_array(x.mapv(|x_| x_ as i32).view())?,
                TensorElementType::Int64 => ort::Value::from_array(x.mapv(|x_| x_ as i64).view())?,
                _ => todo!(),
            };
            xs_.push(x_);
        }
        let t_pre = t_pre.elapsed();

        // inference
        let t_run = std::time::Instant::now();
        let ys = self.session.run(xs_.as_ref())?;
        let t_run = t_run.elapsed();

        // oputput
        let mut ys_ = Vec::new();
        let t_post = std::time::Instant::now();

        for (dtype, name) in self.odtypes.iter().zip(self.onames.iter()) {
            let y = &ys[name.as_str()];
            let y_ = match &dtype {
                TensorElementType::Float32 => y.extract_tensor::<f32>()?.view().to_owned(),
                TensorElementType::Float16 => y.extract_tensor::<f16>()?.view().mapv(f16::to_f32),
                TensorElementType::Int64 => y
                    .extract_tensor::<i64>()?
                    .view()
                    .to_owned()
                    .mapv(|x| x as f32),
                _ => todo!(),
            };
            ys_.push(y_);
        }
        let t_post = t_post.elapsed();
        if self.profile {
            println!(
                "[Profile] batch: {:?} => {:.4?} (i: {t_pre:.4?}, run: {t_run:.4?}, o: {t_post:.4?})", 
                self.batch().opt,
                t_pre + t_run + t_post
            );
        }
        Ok(ys_)
    }

    pub fn _set_ixx(x: isize, ixx: &Option<MinOptMax>, i: usize, ii: usize) -> Option<MinOptMax> {
        match x {
            -1 => {
                match ixx {
                    None => panic!(
                        "{CROSS_MARK} Using dynamic shapes in inputs without specifying it: the {}-th input, the {}-th dimension.",
                        i + 1,
                        ii + 1
                    ),
                    Some(ixx) => Some(ixx.to_owned()), // customized
                }
            }
            _ => Some((x, x, x).into()), // customized, but not dynamic
        }
    }

    pub fn nbytes_from_onnx_dtype_id(x: usize) -> usize {
        match x {
            7 | 11 | 13 => 8,     // i64, f64, u64
            1 | 6 | 12 => 4,      // f32, i32, u32
            10 | 16 | 5 | 4 => 2, // f16, bf16, i16, u16
            2 | 3 | 9 => 1,       // u8, i8, bool
            8 => 4,               // string(1~4)
            _ => todo!(),
        }
    }

    pub fn nbytes_from_onnx_dtype(x: &ort::TensorElementType) -> usize {
        match x {
            ort::TensorElementType::Float64
            | ort::TensorElementType::Uint64
            | ort::TensorElementType::Int64 => 8, // i64, f64, u64
            ort::TensorElementType::Float32
            | ort::TensorElementType::Uint32
            | ort::TensorElementType::Int32
            | ort::TensorElementType::String => 4, // f32, i32, u32, string(1~4)
            ort::TensorElementType::Float16
            | ort::TensorElementType::Bfloat16
            | ort::TensorElementType::Int16
            | ort::TensorElementType::Uint16 => 2, // f16, bf16, i16, u16
            ort::TensorElementType::Uint8
            | ort::TensorElementType::Int8
            | ort::TensorElementType::Bool => 1, // u8, i8, bool
        }
    }

    fn ort_dtype_from_onnx_dtype_id(value: i32) -> Option<ort::TensorElementType> {
        match value {
            0 => None,
            1 => Some(ort::TensorElementType::Float32),
            2 => Some(ort::TensorElementType::Uint8),
            3 => Some(ort::TensorElementType::Int8),
            4 => Some(ort::TensorElementType::Uint16),
            5 => Some(ort::TensorElementType::Int16),
            6 => Some(ort::TensorElementType::Int32),
            7 => Some(ort::TensorElementType::Int64),
            8 => Some(ort::TensorElementType::String),
            9 => Some(ort::TensorElementType::Bool),
            10 => Some(ort::TensorElementType::Float16),
            11 => Some(ort::TensorElementType::Float64),
            12 => Some(ort::TensorElementType::Uint32),
            13 => Some(ort::TensorElementType::Uint64),
            14 => None, // COMPLEX64
            15 => None, // COMPLEX128
            16 => Some(ort::TensorElementType::Bfloat16),
            _ => None,
        }
    }

    #[allow(dead_code)]
    #[allow(clippy::type_complexity)]
    fn parse_io_from_session(
        session: &Session,
    ) -> (
        Vec<Vec<isize>>,
        Vec<ort::TensorElementType>,
        Vec<String>,
        Vec<Vec<isize>>,
        Vec<ort::TensorElementType>,
        Vec<String>,
    ) {
        let mut ishapes = Vec::new();
        let mut idtypes = Vec::new();
        let mut inames = Vec::new();
        for x in session.inputs.iter() {
            inames.push(x.name.to_owned());
            if let ValueType::Tensor { ty, dimensions } = &x.input_type {
                ishapes.push(dimensions.iter().map(|x| *x as isize).collect::<Vec<_>>());
                idtypes.push(*ty);
            } else {
                ishapes.push(vec![-1_isize]);
                idtypes.push(ort::TensorElementType::Float32);
            }
        }

        let mut oshapes = Vec::new();
        let mut odtypes = Vec::new();
        let mut onames = Vec::new();
        for x in session.outputs.iter() {
            onames.push(x.name.to_owned());
            if let ValueType::Tensor { ty, dimensions } = &x.output_type {
                oshapes.push(dimensions.iter().map(|x| *x as isize).collect::<Vec<_>>());
                odtypes.push(*ty);
            } else {
                oshapes.push(vec![-1_isize]);
                odtypes.push(ort::TensorElementType::Float32);
            }
        }

        (ishapes, idtypes, inames, oshapes, odtypes, onames)
    }

    fn parse_onnx_value_info(
        initializer_names: &HashSet<&str>,
        value_info: &[onnx::ValueInfoProto],
    ) -> (Vec<Vec<isize>>, Vec<ort::TensorElementType>, Vec<String>) {
        let mut xshapes: Vec<Vec<isize>> = Vec::new();
        let mut xdtypes: Vec<ort::TensorElementType> = Vec::new();
        let mut xnames: Vec<String> = Vec::new();
        for v in value_info.iter() {
            if initializer_names.contains(v.name.as_str()) {
                continue;
            }
            xnames.push(v.name.to_string());
            let dtype = match &v.r#type {
                Some(dtype) => dtype,
                None => continue,
            };
            let dtype = match &dtype.value {
                Some(dtype) => dtype,
                None => continue,
            };
            let tensor = match dtype {
                onnx::type_proto::Value::TensorType(tensor) => tensor,
                _ => continue,
            };
            let tensor_type = tensor.elem_type;
            let tensor_type = match Self::ort_dtype_from_onnx_dtype_id(tensor_type) {
                Some(dtype) => dtype,
                None => continue,
                // None => anyhow::bail!("DType not supported"),
            };
            xdtypes.push(tensor_type);

            let shapes = match &tensor.shape {
                Some(shapes) => shapes,
                None => continue,
                // None => anyhow::bail!("DType has no shapes"),
            };
            let mut shape_: Vec<isize> = Vec::new();
            for shape in shapes.dim.iter() {
                match &shape.value {
                    None => continue,
                    Some(value) => match value {
                        onnx::tensor_shape_proto::dimension::Value::DimValue(x) => {
                            shape_.push(*x as isize);
                        }
                        onnx::tensor_shape_proto::dimension::Value::DimParam(_) => {
                            shape_.push(-1isize);
                        }
                    },
                }
            }
            xshapes.push(shape_);
        }
        (xshapes, xdtypes, xnames)
    }

    pub fn load_onnx<P: AsRef<std::path::Path>>(p: P) -> Result<onnx::ModelProto> {
        let f = std::fs::read(p)?;
        Ok(onnx::ModelProto::decode(f.as_slice())?)
    }

    pub fn oshapes(&self) -> &Vec<Vec<isize>> {
        &self.oshapes
    }

    pub fn onames(&self) -> &Vec<String> {
        &self.onames
    }

    pub fn odtypes(&self) -> &Vec<ort::TensorElementType> {
        &self.odtypes
    }

    pub fn ishapes(&self) -> &Vec<Vec<isize>> {
        &self.ishapes
    }

    pub fn inames(&self) -> &Vec<String> {
        &self.inames
    }

    pub fn idtypes(&self) -> &Vec<ort::TensorElementType> {
        &self.idtypes
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn inputs_minoptmax(&self) -> &Vec<Vec<MinOptMax>> {
        &self.inputs_minoptmax
    }

    pub fn batch(&self) -> &MinOptMax {
        &self.inputs_minoptmax[0][0]
    }

    pub fn height(&self) -> &MinOptMax {
        &self.inputs_minoptmax[0][2]
    }

    pub fn width(&self) -> &MinOptMax {
        &self.inputs_minoptmax[0][3]
    }

    pub fn is_batch_dyn(&self) -> bool {
        self.ishapes[0][0] == -1
    }

    pub fn try_fetch(&self, key: &str) -> Option<String> {
        match self.session.metadata() {
            Err(_) => None,
            Ok(metadata) => match metadata.custom(key) {
                Err(_) => None,
                Ok(value) => value,
            },
        }
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    pub fn ir_version(&self) -> usize {
        self.model_proto.ir_version as usize
    }

    pub fn opset_version(&self) -> usize {
        self.model_proto.opset_import[0].version as usize
    }

    pub fn producer_name(&self) -> String {
        self.model_proto.producer_name.to_string()
    }

    pub fn producer_version(&self) -> String {
        self.model_proto.producer_version.to_string()
    }

    pub fn model_version(&self) -> usize {
        self.model_proto.model_version as usize
    }

    pub fn parameters(&self) -> usize {
        self.params
    }

    pub fn memory_weights(&self) -> usize {
        self.wbmems
    }
}
