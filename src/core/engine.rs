use anyhow::Result;
use half::f16;
use human_bytes::human_bytes;
use ndarray::{Array, IxDyn};
use ort::{
    ExecutionProvider, Session, SessionBuilder, TensorElementType, TensorRTExecutionProvider,
    MINOR_VERSION,
};
use prost::Message;
use std::collections::HashSet;

use crate::{home_dir, onnx, Device, MinOptMax, Ops, Options, Ts, CHECK_MARK, CROSS_MARK, X};

/// Ort Tensor Attrs: name, data_type, dims
#[derive(Debug)]
pub struct OrtTensorAttr {
    pub names: Vec<String>,
    pub dtypes: Vec<ort::TensorElementType>,
    pub dimss: Vec<Vec<isize>>,
}

/// ONNXRuntime Backend
#[derive(Debug)]
pub struct OrtEngine {
    session: Session,
    device: Device,
    inputs_minoptmax: Vec<Vec<MinOptMax>>,
    inputs_attrs: OrtTensorAttr,
    outputs_attrs: OrtTensorAttr,
    profile: bool,
    num_dry_run: usize,
    model_proto: onnx::ModelProto,
    params: usize,
    wbmems: usize,
    ts: Ts,
}

impl OrtEngine {
    pub fn new(config: &Options) -> Result<Self> {
        // onnx graph
        let model_proto = Self::load_onnx(&config.onnx_path)?;
        let graph = match &model_proto.graph {
            Some(graph) => graph,
            None => anyhow::bail!("No graph found in this proto"),
        };

        // model params & mems
        let byte_alignment = 16; // 16 for simd; 8 for most
        let mut params: usize = 0;
        let mut wbmems: usize = 0;
        let mut initializer_names: HashSet<&str> = HashSet::new();
        for tensor_proto in graph.initializer.iter() {
            initializer_names.insert(&tensor_proto.name);
            let param = tensor_proto.dims.iter().product::<i64>() as usize;
            params += param;

            // mems
            let param = Ops::make_divisible(param, byte_alignment);
            let n = Self::nbytes_from_onnx_dtype_id(tensor_proto.data_type as usize);
            let wbmem = param * n;
            wbmems += wbmem;
        }

        // inputs & outputs
        let inputs_attrs = Self::io_from_onnx_value_info(&initializer_names, &graph.input)?;
        let outputs_attrs = Self::io_from_onnx_value_info(&initializer_names, &graph.output)?;

        // inputs minoptmax
        let mut inputs_minoptmax: Vec<Vec<MinOptMax>> = Vec::new();
        for (i, dims) in inputs_attrs.dimss.iter().enumerate() {
            let mut v_: Vec<MinOptMax> = Vec::new();
            for (ii, &x) in dims.iter().enumerate() {
                let x_default: MinOptMax = (
                    inputs_attrs.dimss[i][ii],
                    inputs_attrs.dimss[i][ii],
                    inputs_attrs.dimss[i][ii],
                )
                    .into();
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

        // build
        ort::init().commit()?;
        let builder = Session::builder()?;
        let mut device = config.device.to_owned();
        match device {
            Device::Trt(device_id) => {
                Self::build_trt(
                    &inputs_attrs.names,
                    &inputs_minoptmax,
                    &builder,
                    device_id,
                    config.trt_int8_enable,
                    config.trt_fp16_enable,
                    config.trt_engine_cache_enable,
                )?;
            }
            Device::Cuda(device_id) => {
                Self::build_cuda(&builder, device_id).unwrap_or_else(|err| {
                    device = Device::Cpu(0);
                    println!("{err}");
                })
            }
            Device::CoreML(_) => Self::build_coreml(&builder).unwrap_or_else(|err| {
                device = Device::Cpu(0);
                println!("{err}");
            }),
            Device::Cpu(_) => {
                Self::build_cpu(&builder)?;
            }
            _ => todo!(),
        }

        let session = builder
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .commit_from_file(&config.onnx_path)?;

        // summary
        println!(
            "{CHECK_MARK} ORT: 1.{MINOR_VERSION}.x | Opset: {} | EP: {:?} | Dtype: {:?} | Parameters: {}",
            model_proto.opset_import[0].version,
            device,
            inputs_attrs.dtypes,
            human_bytes(params as f64),
        );

        Ok(Self {
            session,
            device,
            inputs_minoptmax,
            inputs_attrs,
            outputs_attrs,
            profile: config.profile,
            num_dry_run: config.num_dry_run,
            model_proto,
            params,
            wbmems,
            ts: Ts::default(),
        })
    }

    fn build_trt(
        names: &[String],
        inputs_minoptmax: &[Vec<MinOptMax>],
        builder: &SessionBuilder,
        device_id: usize,
        int8_enable: bool,
        fp16_enable: bool,
        engine_cache_enable: bool,
    ) -> Result<()> {
        // auto generate shapes
        let mut spec_min = String::new();
        let mut spec_opt = String::new();
        let mut spec_max = String::new();
        for (i, name) in names.iter().enumerate() {
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
            .with_profile_max_shapes(spec_max);
        if trt.is_available()? && trt.register(builder).is_ok() {
            println!("\n🐢 Initial model serialization with TensorRT may require a wait...\n");
            Ok(())
        } else {
            anyhow::bail!("{CROSS_MARK} TensorRT initialization failed")
        }
    }

    fn build_cuda(builder: &SessionBuilder, device_id: usize) -> Result<()> {
        let ep = ort::CUDAExecutionProvider::default().with_device_id(device_id as i32);
        if ep.is_available()? && ep.register(builder).is_ok() {
            Ok(())
        } else {
            anyhow::bail!("{CROSS_MARK} CUDA initialization failed")
        }
    }

    fn build_coreml(builder: &SessionBuilder) -> Result<()> {
        let ep = ort::CoreMLExecutionProvider::default().with_subgraphs(); //.with_ane_only();
        if ep.is_available()? && ep.register(builder).is_ok() {
            Ok(())
        } else {
            anyhow::bail!("{CROSS_MARK} CoreML initialization failed")
        }
    }

    fn build_cpu(builder: &SessionBuilder) -> Result<()> {
        let ep = ort::CPUExecutionProvider::default();
        if ep.is_available()? && ep.register(builder).is_ok() {
            Ok(())
        } else {
            anyhow::bail!("{CROSS_MARK} CPU initialization failed")
        }
    }

    pub fn dry_run(&mut self) -> Result<()> {
        if self.num_dry_run > 0 {
            let mut xs = Vec::new();
            for i in self.inputs_minoptmax.iter() {
                let mut x: Vec<usize> = Vec::new();
                for i_ in i.iter() {
                    x.push(i_.opt as usize);
                }
                let x: Array<f32, IxDyn> = Array::ones(x).into_dyn();
                xs.push(X::from(x));
            }
            for _ in 0..self.num_dry_run {
                // self.run(xs.as_ref())?;
                self.run(xs.clone())?;
            }
            self.ts.clear();
            println!("{CHECK_MARK} Dryrun x{}", self.num_dry_run);
        }
        Ok(())
    }

    pub fn run(&mut self, xs: Vec<X>) -> Result<Vec<X>> {
        // inputs dtype alignment
        let mut xs_ = Vec::new();
        let t_pre = std::time::Instant::now();
        for (idtype, x) in self.inputs_attrs.dtypes.iter().zip(xs.iter()) {
            let x_ = match &idtype {
                TensorElementType::Float32 => ort::Value::from_array(x.view())?.into_dyn(),
                TensorElementType::Float16 => {
                    ort::Value::from_array(x.mapv(f16::from_f32).view())?.into_dyn()
                }
                TensorElementType::Int32 => {
                    ort::Value::from_array(x.mapv(|x_| x_ as i32).view())?.into_dyn()
                }
                TensorElementType::Int64 => {
                    ort::Value::from_array(x.mapv(|x_| x_ as i64).view())?.into_dyn()
                }
                _ => todo!(),
            };
            xs_.push(Into::<ort::SessionInputValue<'_>>::into(x_));
        }
        let t_pre = t_pre.elapsed();
        self.ts.add_or_push(0, t_pre);

        // inference
        let t_run = std::time::Instant::now();
        let outputs = self.session.run(&xs_[..])?;
        let t_run = t_run.elapsed();
        self.ts.add_or_push(1, t_run);

        // oputput
        let mut ys = Vec::new();
        let t_post = std::time::Instant::now();
        for (dtype, name) in self
            .outputs_attrs
            .dtypes
            .iter()
            .zip(self.outputs_attrs.names.iter())
        {
            let y = &outputs[name.as_str()];
            let y_ = match &dtype {
                TensorElementType::Float32 => y.try_extract_tensor::<f32>()?.view().into_owned(),
                TensorElementType::Float16 => y
                    .try_extract_tensor::<f16>()?
                    .view()
                    .mapv(f16::to_f32)
                    .into_owned(),
                TensorElementType::Int64 => y
                    .try_extract_tensor::<i64>()?
                    .view()
                    .to_owned()
                    .mapv(|x| x as f32)
                    .into_owned(),
                _ => todo!(),
            };
            // ys.push(y_);
            ys.push(X::from(y_));
        }
        let t_post = t_post.elapsed();
        self.ts.add_or_push(2, t_post);

        if self.profile {
            let len = 10usize;
            let n = 4usize;
            println!(
                "[Profile] {:>len$.n$?} ({:>len$.n$?} avg) [alignment: {:>len$.n$?} ({:>len$.n$?} avg) | inference: {:>len$.n$?} ({:>len$.n$?} avg) | to_f32: {:>len$.n$?} ({:>len$.n$?} avg)]",
                t_pre + t_run + t_post,
                self.ts.avg(),
                t_pre,
                self.ts.avgi(0),
                t_run,
                self.ts.avgi(1),
                t_post,
                self.ts.avgi(2),
            );
        }
        Ok(ys)
    }

    fn _set_ixx(x: isize, ixx: &Option<MinOptMax>, i: usize, ii: usize) -> Option<MinOptMax> {
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

    #[allow(dead_code)]
    fn nbytes_from_onnx_dtype_id(x: usize) -> usize {
        match x {
            7 | 11 | 13 => 8,     // i64, f64, u64
            1 | 6 | 12 => 4,      // f32, i32, u32
            10 | 16 | 5 | 4 => 2, // f16, bf16, i16, u16
            2 | 3 | 9 => 1,       // u8, i8, bool
            8 => 4,               // string(1~4)
            _ => todo!(),
        }
    }

    #[allow(dead_code)]
    fn nbytes_from_onnx_dtype(x: &ort::TensorElementType) -> usize {
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

    #[allow(dead_code)]
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
    fn i_from_session(session: &ort::Session) -> Result<OrtTensorAttr> {
        let mut dimss = Vec::new();
        let mut dtypes = Vec::new();
        let mut names = Vec::new();
        for x in session.inputs.iter() {
            names.push(x.name.to_owned());
            if let ort::ValueType::Tensor { ty, dimensions } = &x.input_type {
                dimss.push(dimensions.iter().map(|x| *x as isize).collect::<Vec<_>>());
                dtypes.push(*ty);
            } else {
                dimss.push(vec![-1_isize]);
                dtypes.push(ort::TensorElementType::Float32);
            }
        }

        Ok(OrtTensorAttr {
            names,
            dimss,
            dtypes,
        })
    }

    #[allow(dead_code)]
    fn o_from_session(session: &ort::Session) -> Result<OrtTensorAttr> {
        let mut dimss = Vec::new();
        let mut dtypes = Vec::new();
        let mut names = Vec::new();
        for x in session.outputs.iter() {
            names.push(x.name.to_owned());
            if let ort::ValueType::Tensor { ty, dimensions } = &x.output_type {
                dimss.push(dimensions.iter().map(|x| *x as isize).collect::<Vec<_>>());
                dtypes.push(*ty);
            } else {
                dimss.push(vec![-1_isize]);
                dtypes.push(ort::TensorElementType::Float32);
            }
        }

        Ok(OrtTensorAttr {
            names,
            dimss,
            dtypes,
        })
    }

    fn io_from_onnx_value_info(
        initializer_names: &HashSet<&str>,
        value_info: &[onnx::ValueInfoProto],
    ) -> Result<OrtTensorAttr> {
        let mut dimss: Vec<Vec<isize>> = Vec::new();
        let mut dtypes: Vec<ort::TensorElementType> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        for v in value_info.iter() {
            if initializer_names.contains(v.name.as_str()) {
                continue;
            }
            names.push(v.name.to_string());
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
            dtypes.push(tensor_type);

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
            dimss.push(shape_);
        }
        Ok(OrtTensorAttr {
            dimss,
            dtypes,
            names,
        })
    }

    pub fn load_onnx<P: AsRef<std::path::Path>>(p: P) -> Result<onnx::ModelProto> {
        let f = std::fs::read(p)?;
        Ok(onnx::ModelProto::decode(f.as_slice())?)
    }

    pub fn oshapes(&self) -> &Vec<Vec<isize>> {
        &self.outputs_attrs.dimss
    }

    pub fn odimss(&self) -> &Vec<Vec<isize>> {
        &self.outputs_attrs.dimss
    }

    pub fn onames(&self) -> &Vec<String> {
        &self.outputs_attrs.names
    }

    pub fn odtypes(&self) -> &Vec<ort::TensorElementType> {
        &self.outputs_attrs.dtypes
    }

    pub fn ishapes(&self) -> &Vec<Vec<isize>> {
        &self.inputs_attrs.dimss
    }

    pub fn idimss(&self) -> &Vec<Vec<isize>> {
        &self.inputs_attrs.dimss
    }

    pub fn inames(&self) -> &Vec<String> {
        &self.inputs_attrs.names
    }

    pub fn idtypes(&self) -> &Vec<ort::TensorElementType> {
        &self.inputs_attrs.dtypes
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
        self.ishapes()[0][0] == -1
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

    pub fn ts(&self) -> &Ts {
        &self.ts
    }
}
