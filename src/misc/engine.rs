use aksr::Builder;
use anyhow::Result;
use half::{bf16, f16};
use log::{error, info, warn};
use ndarray::{Array, IxDyn};
#[allow(unused_imports)]
use ort::{
    execution_providers::ExecutionProvider,
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session, SessionInputValue,
    },
    tensor::TensorElementType,
    value::{DynValue, Value},
};
use prost::Message;
use std::collections::HashSet;

use crate::{
    build_progress_bar, elapsed, human_bytes, onnx, DType, Device, Iiix, MinOptMax, Ops, Ts, Xs, X,
};

/// A struct for tensor attrs composed of the names, the dtypes, and the dimensions.
#[derive(Builder, Debug, Clone)]
pub struct OrtTensorAttr {
    pub names: Vec<String>,
    pub dtypes: Vec<TensorElementType>,
    pub dimss: Vec<Vec<usize>>,
}

#[derive(Debug)]
pub struct OnnxIo {
    pub inputs: OrtTensorAttr,
    pub outputs: OrtTensorAttr,
    pub session: Session,
    pub proto: onnx::ModelProto,
}

#[derive(Debug, Builder)]
pub struct Engine {
    pub file: String,
    pub spec: String,
    pub device: Device,
    pub trt_fp16: bool,
    #[args(inc = true)]
    pub iiixs: Vec<Iiix>,
    #[args(alias = "parameters")]
    pub params: Option<usize>,
    #[args(alias = "memory")]
    pub wbmems: Option<usize>,
    pub inputs_minoptmax: Vec<Vec<MinOptMax>>,
    pub onnx: Option<OnnxIo>,
    pub ts: Ts,
    pub num_dry_run: usize,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            file: Default::default(),
            device: Device::Cpu(0),
            trt_fp16: false,
            spec: Default::default(),
            iiixs: Default::default(),
            num_dry_run: 3,
            params: None,
            wbmems: None,
            inputs_minoptmax: vec![],
            onnx: None,
            ts: Ts::default(),
        }
    }
}

impl Engine {
    pub fn build(mut self) -> Result<Self> {
        let name = format!("[{}] ort_initialization", self.spec);
        elapsed!(&name, self.ts, {
            let proto = Self::load_onnx(self.file())?;
            let graph = match &proto.graph {
                Some(graph) => graph,
                None => {
                    anyhow::bail!(
                        "No graph found in this proto. Invalid ONNX model: {}",
                        self.file()
                    )
                }
            };

            // params & mems
            let byte_alignment = 16;
            let mut params: usize = 0;
            let mut wbmems: usize = 0;
            let mut initializer_names: HashSet<&str> = HashSet::new();
            if !graph.initializer.is_empty() {
                // from initializer
                for tensor_proto in graph.initializer.iter() {
                    initializer_names.insert(&tensor_proto.name);
                    let param = tensor_proto.dims.iter().product::<i64>() as usize;
                    params += param;
                    let param = Ops::make_divisible(param, byte_alignment);
                    let n = Self::nbytes_from_onnx_dtype_id(tensor_proto.data_type as usize);
                    let wbmem = param * n;
                    wbmems += wbmem;
                }
            } else {
                // from node, workaround
                for node in &graph.node {
                    for attr in &node.attribute {
                        if let Some(tensor) = &attr.t {
                            let param = tensor.dims.iter().product::<i64>() as usize;
                            params += param;
                            let param = Ops::make_divisible(param, byte_alignment);
                            let n = Self::nbytes_from_onnx_dtype_id(tensor.data_type as usize);
                            let wbmem = param * n;
                            wbmems += wbmem;
                        }
                    }
                }
            }
            self.params = Some(params);
            self.wbmems = Some(wbmems);

            // inputs & outputs
            let inputs = Self::io_from_onnx_value_info(&initializer_names, &graph.input)?;
            let outputs = Self::io_from_onnx_value_info(&initializer_names, &graph.output)?;
            self.inputs_minoptmax = Self::build_ort_inputs(&inputs, self.iiixs())?;

            // session
            ort::init().commit()?;
            let session = self.build_session(&inputs)?;

            // onnxio
            self.onnx = Some(OnnxIo {
                inputs,
                outputs,
                proto,
                session,
            });
        });
        self.dry_run()?;
        self.info();

        Ok(self)
    }

    pub fn dry_run(&mut self) -> Result<()> {
        if self.num_dry_run > 0 {
            // pb
            let pb = build_progress_bar(
                self.num_dry_run as u64,
                "DryRun",
                Some(self.spec()),
                crate::PROGRESS_BAR_STYLE_CYAN_2,
            )?;

            // dummy
            let mut xs = Vec::new();
            for i in self.inputs_minoptmax().iter() {
                let mut x: Vec<usize> = Vec::new();
                for i_ in i.iter() {
                    x.push(i_.opt());
                }
                let x: Array<f32, IxDyn> = Array::ones(x).into_dyn();
                xs.push(X::from(x));
            }
            let xs = Xs::from(xs);

            // run
            for i in 0..self.num_dry_run {
                pb.inc(1);
                let name = format!("[{}] ort_dry_run_{}", self.spec, i);
                elapsed!(&name, self.ts, {
                    self.run(xs.clone())?;
                });
            }

            // update
            pb.set_message(format!(
                "{}({}) on {:?}",
                self.spec,
                match self.params {
                    Some(bytes) if bytes != 0 => {
                        human_bytes(bytes as f64, true)
                    }
                    _ => "Unknown".to_string(),
                },
                self.device,
            ));
            pb.set_style(indicatif::ProgressStyle::with_template(
                crate::PROGRESS_BAR_STYLE_FINISH,
            )?);
            pb.finish();
        }
        Ok(())
    }

    pub fn run(&mut self, xs: Xs) -> Result<Xs> {
        let mut ys = xs.derive();
        if let Some(onnx) = &self.onnx {
            // alignment
            let xs_ = elapsed!(&format!("[{}] ort_preprocessing", self.spec), self.ts, {
                let mut xs_ = Vec::new();
                for (dtype, x) in onnx.inputs.dtypes.iter().zip(xs.into_iter()) {
                    xs_.push(Into::<SessionInputValue<'_>>::into(Self::preprocess(
                        x, dtype,
                    )?));
                }

                xs_
            });

            // run
            let outputs = elapsed!(
                &format!("[{}] ort_inference", self.spec),
                self.ts,
                onnx.session.run(&xs_[..])?
            );

            // extract
            elapsed!(&format!("[{}] ort_postprocessing", self.spec), self.ts, {
                for (dtype, name) in onnx.outputs.dtypes.iter().zip(onnx.outputs.names.iter()) {
                    let y = Self::postprocess(&outputs[name.as_str()], dtype)?;
                    ys.push_kv(name.as_str(), X::from(y))?;
                }
            });

            Ok(ys)
        } else {
            anyhow::bail!("Failed to run with ONNXRuntime. No model info found.");
        }
    }

    fn preprocess(x: &X, dtype: &TensorElementType) -> Result<DynValue> {
        let x = match dtype {
            TensorElementType::Float32 => Value::from_array(x.view())?.into_dyn(),
            TensorElementType::Float16 => {
                Value::from_array(x.mapv(f16::from_f32).view())?.into_dyn()
            }
            TensorElementType::Float64 => Value::from_array(x.view())?.into_dyn(),
            TensorElementType::Bfloat16 => {
                Value::from_array(x.mapv(bf16::from_f32).view())?.into_dyn()
            }
            TensorElementType::Int8 => Value::from_array(x.mapv(|x_| x_ as i8).view())?.into_dyn(),
            TensorElementType::Int16 => {
                Value::from_array(x.mapv(|x_| x_ as i16).view())?.into_dyn()
            }
            TensorElementType::Int32 => {
                Value::from_array(x.mapv(|x_| x_ as i32).view())?.into_dyn()
            }
            TensorElementType::Int64 => {
                Value::from_array(x.mapv(|x_| x_ as i64).view())?.into_dyn()
            }
            TensorElementType::Uint8 => Value::from_array(x.mapv(|x_| x_ as u8).view())?.into_dyn(),
            TensorElementType::Uint16 => {
                Value::from_array(x.mapv(|x_| x_ as u16).view())?.into_dyn()
            }
            TensorElementType::Uint32 => {
                Value::from_array(x.mapv(|x_| x_ as u32).view())?.into_dyn()
            }
            TensorElementType::Uint64 => {
                Value::from_array(x.mapv(|x_| x_ as u64).view())?.into_dyn()
            }
            TensorElementType::Bool => Value::from_array(x.mapv(|x_| x_ != 0.).view())?.into_dyn(),
            _ => unimplemented!(),
        };

        Ok(x)
    }

    fn postprocess(x: &DynValue, dtype: &TensorElementType) -> Result<Array<f32, IxDyn>> {
        fn _extract_and_convert<T>(x: &DynValue, map_fn: impl Fn(T) -> f32) -> Array<f32, IxDyn>
        where
            T: Clone + 'static + ort::tensor::PrimitiveTensorElementType,
        {
            match x.try_extract_tensor::<T>() {
                Err(err) => {
                    error!("Failed to extract from ort outputs: {:?}", err);
                    Array::zeros(0).into_dyn()
                }
                Ok(x) => x.view().mapv(map_fn).into_owned(),
            }
        }
        let x = match dtype {
            TensorElementType::Float32 => _extract_and_convert::<f32>(x, |x| x),
            TensorElementType::Float16 => _extract_and_convert::<f16>(x, f16::to_f32),
            TensorElementType::Bfloat16 => _extract_and_convert::<bf16>(x, bf16::to_f32),
            TensorElementType::Float64 => _extract_and_convert::<f64>(x, |x| x as f32),
            TensorElementType::Int64 => _extract_and_convert::<i64>(x, |x| x as f32),
            TensorElementType::Int32 => _extract_and_convert::<i32>(x, |x| x as f32),
            TensorElementType::Int16 => _extract_and_convert::<i16>(x, |x| x as f32),
            TensorElementType::Int8 => _extract_and_convert::<i8>(x, |x| x as f32),
            TensorElementType::Uint64 => _extract_and_convert::<u64>(x, |x| x as f32),
            TensorElementType::Uint32 => _extract_and_convert::<u32>(x, |x| x as f32),
            TensorElementType::Uint16 => _extract_and_convert::<u16>(x, |x| x as f32),
            TensorElementType::Uint8 => _extract_and_convert::<u8>(x, |x| x as f32),
            TensorElementType::Bool => _extract_and_convert::<bool>(x, |x| x as u8 as f32),
            _ => return Err(anyhow::anyhow!("Unsupported ort tensor type: {:?}", dtype)),
        };

        Ok(x)
    }

    #[allow(unused_variables)]
    fn build_session(&mut self, inputs: &OrtTensorAttr) -> Result<Session> {
        #[allow(unused_mut)]
        let mut builder = Session::builder()?;
        let compile_help = "Please compile ONNXRuntime with #EP";
        let feature_help = "#EP EP requires the features: `#FEATURE`. \
            \nConsider enabling them by passing, e.g., `--features #FEATURE`";

        match self.device {
            Device::TensorRT(id) => {
                #[cfg(not(feature = "trt"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "TensorRT")
                        .replace("#FEATURE", "trt"));
                }

                #[cfg(feature = "trt")]
                {
                    // generate shapes
                    let mut spec_min = String::new();
                    let mut spec_opt = String::new();
                    let mut spec_max = String::new();
                    for (i, name) in inputs.names.iter().enumerate() {
                        if i != 0 {
                            spec_min.push(',');
                            spec_opt.push(',');
                            spec_max.push(',');
                        }
                        let mut s_min = format!("{}:", name);
                        let mut s_opt = format!("{}:", name);
                        let mut s_max = format!("{}:", name);
                        for d in self.inputs_minoptmax[i].iter() {
                            let min_ = &format!("{}x", d.min());
                            let opt_ = &format!("{}x", d.opt());
                            let max_ = &format!("{}x", d.max());
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

                    let p = crate::Dir::Cache.path_with_subs(&["trt-cache"])?;
                    let ep = ort::execution_providers::TensorRTExecutionProvider::default()
                        .with_device_id(id as i32)
                        .with_fp16(self.trt_fp16)
                        .with_engine_cache(true)
                        .with_engine_cache_path(p.to_str().unwrap())
                        .with_timing_cache(false)
                        .with_profile_min_shapes(spec_min)
                        .with_profile_opt_shapes(spec_opt)
                        .with_profile_max_shapes(spec_max);

                    match ep.is_available() {
                        Ok(true) => {
                            info!(
                                "Initial model serialization with TensorRT may require a wait..."
                            );
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register TensorRT: {}", err)
                            })?;
                        }
                        _ => {
                            anyhow::bail!(compile_help.replace("#EP", "TensorRT"))
                        }
                    }
                }
            }
            Device::Cuda(id) => {
                #[cfg(not(feature = "cuda"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "CUDA")
                        .replace("#FEATURE", "cuda"));
                }

                #[cfg(feature = "cuda")]
                {
                    let ep = ort::execution_providers::CUDAExecutionProvider::default()
                        .with_device_id(id as i32);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register CUDA: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "CUDA")),
                    }
                }
            }
            Device::CoreML(id) => {
                #[cfg(not(feature = "mps"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "CoreML")
                        .replace("#FEATURE", "mps"));
                }
                #[cfg(feature = "mps")]
                {
                    let ep = ort::execution_providers::CoreMLExecutionProvider::default();
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register CoreML: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "CoreML")),
                    }
                }
            }
            _ => {
                let ep = ort::execution_providers::CPUExecutionProvider::default();
                match ep.is_available() {
                    Ok(true) => {
                        ep.register(&mut builder)
                            .map_err(|err| anyhow::anyhow!("Failed to register Cpu: {}", err))?;
                    }
                    _ => anyhow::bail!(compile_help.replace("#EP", "Cpu")),
                }
            }
        }

        // session
        let session = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(std::thread::available_parallelism()?.get())?
            .commit_from_file(self.file())?;

        Ok(session)
    }

    fn build_ort_inputs(xs: &OrtTensorAttr, iiixs: &[Iiix]) -> Result<Vec<Vec<MinOptMax>>> {
        // init
        let mut ys: Vec<Vec<MinOptMax>> = xs
            .dimss
            .iter()
            .map(|dims| dims.iter().map(|&x| MinOptMax::from(x)).collect())
            .collect();

        // update from customized
        for iiix in iiixs.iter() {
            if let Some(x) = xs.dimss.get(iiix.i).and_then(|dims| dims.get(iiix.ii)) {
                // dynamic
                if *x == 0 {
                    ys[iiix.i][iiix.ii] = iiix.x.clone();
                }
            } else {
                anyhow::bail!(
                    "Cannot retrieve the {}-th dimension of the {}-th input.",
                    iiix.ii,
                    iiix.i,
                );
            }
        }

        // set batch size <- i00
        let batch_size: MinOptMax = if ys[0][0].is_dyn() {
            1.into()
        } else {
            ys[0][0].clone()
        };

        // deal with the dynamic axis
        ys.iter_mut().enumerate().for_each(|(i, xs)| {
            xs.iter_mut().enumerate().for_each(|(ii, x)| {
                if x.is_dyn() {
                    let z = if ii == 0 {
                        batch_size.clone()
                    } else {
                        let z =  MinOptMax::from(1);
                        warn!(
                            "Using dynamic shapes in inputs without specifying it: the {}-th input, the {}-th dimension. \
                            Using {:?} by default. You should make it clear when using TensorRT.",
                            i + 1, ii + 1, z
                        );
                        z
                    };
                    *x = z;
                }
            });
        });

        Ok(ys)
    }

    #[allow(dead_code)]
    fn nbytes_from_onnx_dtype_id(x: usize) -> usize {
        match x {
            7 | 11 | 13 => 8,     // i64, f64, u64
            1 | 6 | 12 => 4,      // f32, i32, u32
            10 | 16 | 5 | 4 => 2, // f16, bf16, i16, u16
            2 | 3 | 9 => 1,       // u8, i8, bool
            8 => 4,               // string(1~4)
            _ => 1,               // TODO: others
        }
    }

    #[allow(dead_code)]
    fn nbytes_from_onnx_dtype(x: &TensorElementType) -> usize {
        match x {
            TensorElementType::Float64 | TensorElementType::Uint64 | TensorElementType::Int64 => 8, // i64, f64, u64
            TensorElementType::Float32
            | TensorElementType::Uint32
            | TensorElementType::Int32
            | TensorElementType::String => 4, // f32, i32, u32, string(1~4)
            TensorElementType::Float16
            | TensorElementType::Bfloat16
            | TensorElementType::Int16
            | TensorElementType::Uint16 => 2, // f16, bf16, i16, u16
            TensorElementType::Uint8 | TensorElementType::Int8 | TensorElementType::Bool => 1, // u8, i8, bool
        }
    }

    #[allow(dead_code)]
    fn ort_dtype_from_onnx_dtype_id(value: i32) -> Option<TensorElementType> {
        match value {
            0 => None,
            1 => Some(TensorElementType::Float32),
            2 => Some(TensorElementType::Uint8),
            3 => Some(TensorElementType::Int8),
            4 => Some(TensorElementType::Uint16),
            5 => Some(TensorElementType::Int16),
            6 => Some(TensorElementType::Int32),
            7 => Some(TensorElementType::Int64),
            8 => Some(TensorElementType::String),
            9 => Some(TensorElementType::Bool),
            10 => Some(TensorElementType::Float16),
            11 => Some(TensorElementType::Float64),
            12 => Some(TensorElementType::Uint32),
            13 => Some(TensorElementType::Uint64),
            14 => None, // COMPLEX64
            15 => None, // COMPLEX128
            16 => Some(TensorElementType::Bfloat16),
            _ => None,
        }
    }

    fn io_from_onnx_value_info(
        initializer_names: &HashSet<&str>,
        value_info: &[onnx::ValueInfoProto],
    ) -> Result<OrtTensorAttr> {
        let mut dimss: Vec<Vec<usize>> = Vec::new();
        let mut dtypes: Vec<TensorElementType> = Vec::new();
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
            };
            dtypes.push(tensor_type);

            let shapes = match &tensor.shape {
                Some(shapes) => shapes,
                None => continue,
            };
            let mut shape_: Vec<usize> = Vec::new();
            for shape in shapes.dim.iter() {
                match &shape.value {
                    None => continue,
                    Some(value) => match value {
                        onnx::tensor_shape_proto::dimension::Value::DimValue(x) => {
                            shape_.push(*x as _);
                        }
                        onnx::tensor_shape_proto::dimension::Value::DimParam(_) => {
                            shape_.push(0);
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
        let f = std::fs::read(p.as_ref())?;
        onnx::ModelProto::decode(f.as_slice()).map_err(|err| {
            anyhow::anyhow!(
                "Failed to read the ONNX model: The file might be incomplete or corrupted. More detailed: {}",
                err
            )
        })
    }

    pub fn batch(&self) -> &MinOptMax {
        &self.inputs_minoptmax[0][0]
    }

    pub fn is_batch_dyn(&self) -> bool {
        self.batch().is_dyn()
    }

    pub fn try_height(&self) -> Option<&MinOptMax> {
        self.inputs_minoptmax.first().and_then(|x| x.get(2))
    }

    pub fn height(&self) -> &MinOptMax {
        // unsafe
        &self.inputs_minoptmax[0][2]
    }

    pub fn is_height_dyn(&self) -> bool {
        self.height().is_dyn()
    }

    pub fn try_width(&self) -> Option<&MinOptMax> {
        self.inputs_minoptmax.first().and_then(|x| x.get(3))
    }

    pub fn width(&self) -> &MinOptMax {
        // unsafe
        &self.inputs_minoptmax[0][3]
    }

    pub fn is_width_dyn(&self) -> bool {
        self.width().is_dyn()
    }

    pub fn try_fetch(&self, key: &str) -> Option<String> {
        match self.onnx.as_ref().unwrap().session.metadata() {
            Err(_) => None,
            Ok(metadata) => metadata.custom(key).unwrap_or_default(),
        }
    }

    pub fn ir_version(&self) -> Option<usize> {
        self.onnx.as_ref().map(|x| x.proto.ir_version as usize)
    }

    pub fn opset_version(&self) -> Option<usize> {
        self.onnx
            .as_ref()
            .map(|x| x.proto.opset_import[0].version as usize)
    }

    pub fn producer_name(&self) -> Option<String> {
        self.onnx.as_ref().map(|x| x.proto.producer_name.clone())
    }

    pub fn producer_version(&self) -> Option<String> {
        self.onnx.as_ref().map(|x| x.proto.producer_version.clone())
    }

    pub fn model_version(&self) -> Option<usize> {
        self.onnx.as_ref().map(|x| x.proto.model_version as usize)
    }

    pub fn ishapes(&self) -> Option<&[Vec<usize>]> {
        self.onnx.as_ref().map(|x| x.inputs.dimss())
    }

    pub fn idimss(&self) -> Option<&[Vec<usize>]> {
        self.onnx.as_ref().map(|x| x.inputs.dimss())
    }

    pub fn inames(&self) -> Option<&[String]> {
        self.onnx.as_ref().map(|x| x.inputs.names())
    }

    pub fn idtypes(&self) -> Option<Vec<DType>> {
        self.onnx.as_ref().and_then(|x| {
            x.inputs
                .dtypes()
                .iter()
                .map(DType::from_ort)
                .collect::<Vec<DType>>()
                .into()
        })
    }

    pub fn oshapes(&self) -> Option<&[Vec<usize>]> {
        self.onnx.as_ref().map(|x| x.outputs.dimss())
    }

    pub fn odimss(&self) -> Option<&[Vec<usize>]> {
        self.onnx.as_ref().map(|x| x.outputs.dimss())
    }

    pub fn onames(&self) -> Option<&[String]> {
        self.onnx.as_ref().map(|x| x.outputs.names())
    }

    pub fn odtypes(&self) -> Option<Vec<DType>> {
        self.onnx.as_ref().and_then(|x| {
            x.outputs
                .dtypes()
                .iter()
                .map(DType::from_ort)
                .collect::<Vec<DType>>()
                .into()
        })
    }

    pub fn profile(&self) {
        self.ts.summary();
    }

    pub fn info(&self) {
        let info = format!(
            "Minimum Supported Ort Version: 1.{}.x, Opset Version: {}, Device: {}, Parameters: {}, Memory: {}",
            ort::MINOR_VERSION,
            self.opset_version().map_or("Unknown".to_string(), |x| x.to_string()),
            self.device,
            match self.params {
                Some(bytes) if bytes != 0 => {
                    human_bytes(bytes as f64, true)
                }
                _ => "Unknown".to_string(),
            },
            match self.wbmems {
                Some(bytes) if bytes != 0 => {
                    human_bytes(bytes as f64, true)
                }
                _ => "Unknown".to_string(),
            },
        );

        info!("{}", info);
    }
}
