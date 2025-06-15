use aksr::Builder;
use anyhow::Result;
use half::{bf16, f16};
use log::{debug, info, warn};
use ndarray::{Array, IxDyn};
use ort::{
    execution_providers::ExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session, SessionInputValue},
    tensor::TensorElementType,
    value::{DynValue, Value},
};
use prost::Message;
use std::collections::HashSet;

use crate::{
    build_progress_bar, elapsed_global, human_bytes_binary, onnx, DType, Device, HardwareConfig,
    Iiix, MinOptMax, ORTConfig, Ops, Xs, PROGRESS_BAR_STYLE_CYAN_2, PROGRESS_BAR_STYLE_FINISH, X,
};

impl From<TensorElementType> for DType {
    fn from(dtype: TensorElementType) -> Self {
        match dtype {
            TensorElementType::Int4 => Self::Int4,
            TensorElementType::Int8 => Self::Int8,
            TensorElementType::Int16 => Self::Int16,
            TensorElementType::Int32 => Self::Int32,
            TensorElementType::Int64 => Self::Int64,
            TensorElementType::Uint4 => Self::Uint4,
            TensorElementType::Uint8 => Self::Uint8,
            TensorElementType::Uint16 => Self::Uint16,
            TensorElementType::Uint32 => Self::Uint32,
            TensorElementType::Uint64 => Self::Uint64,
            TensorElementType::Float16 => Self::Fp16,
            TensorElementType::Float32 => Self::Fp32,
            TensorElementType::Float64 => Self::Fp64,
            TensorElementType::Bfloat16 => Self::Bf16,
            TensorElementType::Float8E4M3FN => Self::Fp8e4m3fn,
            TensorElementType::Float8E4M3FNUZ => Self::Fp8e4m3fnuz,
            TensorElementType::Float8E5M2 => Self::Fp8e5m2,
            TensorElementType::Float8E5M2FNUZ => Self::Fp8e5m2fnuz,
            TensorElementType::Complex64 => Self::Complex64,
            TensorElementType::Complex128 => Self::Complex128,
            _ => todo!(),
        }
    }
}

/// A struct for tensor attrs composed of the names, the dtypes, and the dimensions.
/// ONNX Runtime tensor attributes containing names, data types, and dimensions.
#[derive(Builder, Debug, Clone, Default)]
/// ONNX Runtime tensor attributes containing metadata.
pub struct OrtTensorAttr {
    /// Tensor names.
    pub names: Vec<String>,
    /// Tensor data types.
    pub dtypes: Vec<TensorElementType>,
    /// Tensor dimensions for each tensor.
    pub dimss: Vec<Vec<usize>>,
}

/// ONNX I/O structure containing input/output attributes and session.
#[derive(Debug)]
pub struct OnnxIo {
    /// Input tensor attributes.
    pub inputs: OrtTensorAttr,
    /// Output tensor attributes.
    pub outputs: OrtTensorAttr,
    /// ONNX Runtime session.
    pub session: Session,
    /// ONNX model protocol buffer.
    pub proto: onnx::ModelProto,
}

/// ONNX Runtime inference engine with configuration and session management.
#[derive(Debug, Builder)]
pub struct Engine {
    /// Model file path.
    pub file: String,
    /// Model specification string.
    pub spec: String,
    /// Execution device.
    pub device: Device,
    #[args(inc)]
    pub iiixs: Vec<Iiix>,
    #[args(aka = "parameters")]
    pub params: Option<usize>,
    #[args(aka = "memory")]
    pub wbmems: Option<usize>,
    /// Input min-opt-max configurations.
    pub inputs_minoptmax: Vec<Vec<MinOptMax>>,
    /// ONNX I/O structure.
    pub onnx: Option<OnnxIo>,
    /// Number of dry runs for warmup.
    pub num_dry_run: usize,

    // global
    pub graph_opt_level: Option<u8>,
    pub num_intra_threads: Option<usize>,
    pub num_inter_threads: Option<usize>,

    /// Hardware-specific configurations for all execution providers
    pub hardware: HardwareConfig,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            file: Default::default(),
            device: Device::Cpu(0),
            spec: Default::default(),
            iiixs: Default::default(),
            num_dry_run: 3,
            params: None,
            wbmems: None,
            inputs_minoptmax: vec![],
            onnx: None,
            // global
            graph_opt_level: None,
            num_intra_threads: None,
            num_inter_threads: None,
            // hardware configurations
            hardware: HardwareConfig::new(),
        }
    }
}

impl Engine {
    pub fn try_from_config(config: &ORTConfig) -> Result<Self> {
        let hardware = HardwareConfig {
            cpu: crate::CpuConfig {
                arena_allocator: config.hardware.cpu.arena_allocator,
            },
            tensorrt: crate::TensorRtConfig {
                fp16: config.hardware.tensorrt.fp16,
                engine_cache: config.hardware.tensorrt.engine_cache,
                timing_cache: config.hardware.tensorrt.timing_cache,
            },
            openvino: crate::OpenVinoConfig {
                dynamic_shapes: config.hardware.openvino.dynamic_shapes,
                opencl_throttling: config.hardware.openvino.opencl_throttling,
                qdq_optimizer: config.hardware.openvino.qdq_optimizer,
                num_threads: config.hardware.openvino.num_threads,
            },
            onednn: crate::OneDnnConfig {
                arena_allocator: config.hardware.onednn.arena_allocator,
            },
            coreml: crate::CoreMlConfig {
                static_input_shapes: config.hardware.coreml.static_input_shapes,
                subgraph_running: config.hardware.coreml.subgraph_running,
            },
            cann: crate::CannConfig {
                graph_inference: config.hardware.cann.graph_inference,
                dump_graphs: config.hardware.cann.dump_graphs,
                dump_om_model: config.hardware.cann.dump_om_model,
            },
            nnapi: crate::NnapiConfig {
                cpu_only: config.hardware.nnapi.cpu_only,
                disable_cpu: config.hardware.nnapi.disable_cpu,
                fp16: config.hardware.nnapi.fp16,
                nchw: config.hardware.nnapi.nchw,
            },
            armnn: crate::ArmNnConfig {
                arena_allocator: config.hardware.armnn.arena_allocator,
            },
            migraphx: crate::MiGraphXConfig {
                fp16: config.hardware.migraphx.fp16,
                exhaustive_tune: config.hardware.migraphx.exhaustive_tune,
            },
        };

        Self {
            file: config.file.clone(),
            spec: config.spec.clone(),
            iiixs: config.iiixs.clone(),
            device: config.device,
            num_dry_run: config.num_dry_run,
            // global
            graph_opt_level: config.graph_opt_level,
            num_intra_threads: config.num_intra_threads,
            num_inter_threads: config.num_inter_threads,
            // hardware configurations
            hardware,
            ..Default::default()
        }
        .build()
    }

    pub fn build(mut self) -> Result<Self> {
        let name = format!("[{}] ort_initialization", self.spec);
        elapsed_global!(&name, {
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
                    let n = Self::get_ort_dtype_from_proto_dtype_id(tensor_proto.data_type)
                        .map(|x| x.byte_size(1))
                        .unwrap_or_default();
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
                            let n = Self::get_ort_dtype_from_proto_dtype_id(tensor.data_type)
                                .map(|x| x.byte_size(1))
                                .unwrap_or_default();

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
                PROGRESS_BAR_STYLE_CYAN_2,
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
                elapsed_global!(&name, {
                    self.run(xs.clone())?;
                });
            }

            // update
            pb.set_message(format!(
                "{}({}) on {}",
                self.spec,
                match self.params {
                    Some(bytes) if bytes != 0 => {
                        human_bytes_binary(bytes as f64, 2)
                    }
                    _ => "Unknown".to_string(),
                },
                self.device,
            ));
            pb.set_style(indicatif::ProgressStyle::with_template(
                PROGRESS_BAR_STYLE_FINISH,
            )?);
            pb.finish();
        }
        Ok(())
    }

    pub fn run(&mut self, xs: Xs) -> Result<Xs> {
        let mut ys = xs.derive();
        if let Some(onnx) = &mut self.onnx {
            // alignment
            let xs_ = elapsed_global!(&format!("[{}] ort_preprocessing", self.spec), {
                let mut xs_ = Vec::new();
                for (dtype, x) in onnx.inputs.dtypes.iter().zip(xs.into_iter()) {
                    xs_.push(Into::<SessionInputValue<'_>>::into(Self::preprocess(
                        x, dtype,
                    )?));
                }

                xs_
            });

            // run
            let outputs = elapsed_global!(
                &format!("[{}] ort_inference", self.spec),
                onnx.session.run(&xs_[..])?
            );

            // extract
            elapsed_global!(&format!("[{}] ort_postprocessing", self.spec), {
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
            TensorElementType::Float32 | TensorElementType::Float64 => {
                Value::from_array(x.0.clone())?.into_dyn()
            }
            TensorElementType::Float16 => Value::from_array(x.mapv(f16::from_f32))?.into_dyn(),
            TensorElementType::Bfloat16 => Value::from_array(x.mapv(bf16::from_f32))?.into_dyn(),
            TensorElementType::Int8 => Value::from_array(x.mapv(|x_| x_ as i8))?.into_dyn(),
            TensorElementType::Int16 => Value::from_array(x.mapv(|x_| x_ as i16))?.into_dyn(),
            TensorElementType::Int32 => Value::from_array(x.mapv(|x_| x_ as i32))?.into_dyn(),
            TensorElementType::Int64 => Value::from_array(x.mapv(|x_| x_ as i64))?.into_dyn(),
            TensorElementType::Uint8 => Value::from_array(x.mapv(|x_| x_ as u8))?.into_dyn(),
            TensorElementType::Uint16 => Value::from_array(x.mapv(|x_| x_ as u16))?.into_dyn(),
            TensorElementType::Uint32 => Value::from_array(x.mapv(|x_| x_ as u32))?.into_dyn(),
            TensorElementType::Uint64 => Value::from_array(x.mapv(|x_| x_ as u64))?.into_dyn(),
            TensorElementType::Bool => Value::from_array(x.mapv(|x_| x_ != 0.))?.into_dyn(),
            _ => unimplemented!(),
        };
        Ok(x)
    }

    fn postprocess(x: &DynValue, dtype: &TensorElementType) -> Result<Array<f32, IxDyn>> {
        fn _extract_and_convert<T>(x: &DynValue, map_fn: impl Fn(T) -> f32) -> Array<f32, IxDyn>
        where
            T: Clone + 'static + ort::tensor::PrimitiveTensorElementType,
        {
            match x.try_extract_array::<T>() {
                Err(err) => {
                    debug!("Failed to extract from ort outputs: {:?}. A default value has been generated.", err);
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
        let n_threads_available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        match self.device {
            Device::TensorRt(id) => {
                #[cfg(not(feature = "tensorrt"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "TensorRT")
                        .replace("#FEATURE", "tensorrt"));
                }

                #[cfg(feature = "tensorrt")]
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

                    let ep = ort::execution_providers::TensorRTExecutionProvider::default()
                        .with_device_id(id as i32)
                        .with_fp16(self.hardware.tensorrt.fp16)
                        .with_engine_cache(self.hardware.tensorrt.engine_cache)
                        .with_timing_cache(self.hardware.tensorrt.timing_cache)
                        .with_engine_cache_path(
                            crate::Dir::Cache
                                .crate_dir_default_with_subs(&["caches", "tensorrt"])?
                                .display(),
                        )
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
            Device::CoreMl => {
                #[cfg(not(feature = "coreml"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "CoreML")
                        .replace("#FEATURE", "coreml"));
                }
                #[cfg(feature = "coreml")]
                {
                    let ep = ort::execution_providers::CoreMLExecutionProvider::default()
                        .with_model_cache_dir(
                            crate::Dir::Cache
                                .crate_dir_default_with_subs(&["caches", "coreml"])?
                                .display(),
                        )
                        .with_static_input_shapes(self.hardware.coreml.static_input_shapes)
                .with_subgraphs(self.hardware.coreml.subgraph_running)
                        .with_compute_units(ort::execution_providers::coreml::CoreMLComputeUnits::All)
                        .with_model_format(ort::execution_providers::coreml::CoreMLModelFormat::MLProgram)
                        .with_specialization_strategy(
                            ort::execution_providers::coreml::CoreMLSpecializationStrategy::FastPrediction,
                        );
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
            Device::OpenVino(dt) => {
                #[cfg(not(feature = "openvino"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "OpenVINO")
                        .replace("#FEATURE", "openvino"));
                }

                #[cfg(feature = "openvino")]
                {
                    let ep = ort::execution_providers::OpenVINOExecutionProvider::default()
                        .with_device_type(dt)
                        .with_num_threads(
                            self.hardware
                                .openvino
                                .num_threads
                                .unwrap_or(n_threads_available),
                        )
                        .with_dynamic_shapes(self.hardware.openvino.dynamic_shapes)
                        .with_opencl_throttling(self.hardware.openvino.opencl_throttling)
                        .with_qdq_optimizer(self.hardware.openvino.qdq_optimizer)
                        .with_cache_dir(
                            crate::Dir::Cache
                                .crate_dir_default_with_subs(&["caches", "openvino"])?
                                .display()
                                .to_string(),
                        );
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register OpenVINO: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "OpenVINO")),
                    }
                }
            }
            Device::DirectMl(id) => {
                #[cfg(not(feature = "directml"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "DirectML")
                        .replace("#FEATURE", "directml"));
                }
                #[cfg(feature = "directml")]
                {
                    let ep = ort::execution_providers::DirectMLExecutionProvider::default()
                        .with_device_id(id as i32);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register DirectML: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "DirectML")),
                    }
                }
            }
            Device::Xnnpack => {
                #[cfg(not(feature = "xnnpack"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "XNNPack")
                        .replace("#FEATURE", "xnnpack"));
                }
                #[cfg(feature = "xnnpack")]
                {
                    let ep = ort::execution_providers::XNNPACKExecutionProvider::default();
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register XNNPack: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "XNNPack")),
                    }
                }
            }
            Device::Cann(id) => {
                #[cfg(not(feature = "cann"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "CANN")
                        .replace("#FEATURE", "cann"));
                }
                #[cfg(feature = "cann")]
                {
                    let ep = ort::execution_providers::CANNExecutionProvider::default()
                        .with_device_id(id as i32)
                        .with_cann_graph(self.hardware.cann.graph_inference)
                        .with_dump_graphs(self.hardware.cann.dump_graphs)
                        .with_dump_om_model(self.hardware.cann.dump_om_model);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register CANN: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "CANN")),
                    }
                }
            }
            Device::RkNpu => {
                #[cfg(not(feature = "rknpu"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "RKNPU")
                        .replace("#FEATURE", "rknpu"));
                }
                #[cfg(feature = "rknpu")]
                {
                    let ep = ort::execution_providers::RKNPUExecutionProvider::default();
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register RKNPU: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "RKNPU")),
                    }
                }
            }
            Device::OneDnn => {
                #[cfg(not(feature = "onednn"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "oneDNN")
                        .replace("#FEATURE", "onednn"));
                }
                #[cfg(feature = "onednn")]
                {
                    let ep = ort::execution_providers::OneDNNExecutionProvider::default()
                        .with_arena_allocator(self.hardware.onednn.arena_allocator);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register oneDNN: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "oneDNN")),
                    }
                }
            }
            Device::Acl => {
                #[cfg(not(feature = "acl"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "ArmACL")
                        .replace("#FEATURE", "acl"));
                }
                #[cfg(feature = "acl")]
                {
                    let ep = ort::execution_providers::ACLExecutionProvider::default()
                        .with_fast_math(true);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register ArmACL: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "ArmACL")),
                    }
                }
            }
            Device::Rocm(id) => {
                #[cfg(not(feature = "rocm"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "ROCm")
                        .replace("#FEATURE", "rocm"));
                }
                #[cfg(feature = "rocm")]
                {
                    let ep = ort::execution_providers::ROCmExecutionProvider::default()
                        .with_device_id(id as _);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register ROCm: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "ROCm")),
                    }
                }
            }
            Device::NnApi => {
                #[cfg(not(feature = "nnapi"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "NNAPI")
                        .replace("#FEATURE", "nnapi"));
                }
                #[cfg(feature = "nnapi")]
                {
                    let ep = ort::execution_providers::NNAPIExecutionProvider::default()
                        .with_fp16(self.hardware.nnapi.fp16)
                        .with_nchw(self.hardware.nnapi.nchw)
                        .with_cpu_only(self.hardware.nnapi.cpu_only)
                        .with_disable_cpu(self.hardware.nnapi.disable_cpu);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register NNAPI: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "NNAPI")),
                    }
                }
            }
            Device::ArmNn => {
                #[cfg(not(feature = "armnn"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "ArmNN")
                        .replace("#FEATURE", "armnn"));
                }
                #[cfg(feature = "armnn")]
                {
                    let ep = ort::execution_providers::ArmNNExecutionProvider::default()
                        .with_arena_allocator(self.hardware.armnn.arena_allocator);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register ArmNN: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "ArmNN")),
                    }
                }
            }
            Device::Tvm => {
                #[cfg(not(feature = "tvm"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "TVM")
                        .replace("#FEATURE", "tvm"));
                }
                #[cfg(feature = "tvm")]
                {
                    let ep = ort::execution_providers::TVMExecutionProvider::default();
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register TVM: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "TVM")),
                    }
                }
            }
            Device::Qnn(id) => {
                #[cfg(not(feature = "qnn"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "QNN")
                        .replace("#FEATURE", "qnn"));
                }
                #[cfg(feature = "qnn")]
                {
                    let ep = ort::execution_providers::QNNExecutionProvider::default()
                        .with_device_id(id as _);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register QNN: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "QNN")),
                    }
                }
            }
            Device::MiGraphX(id) => {
                #[cfg(not(feature = "migraphx"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "MIGraphX")
                        .replace("#FEATURE", "migraphx"));
                }
                #[cfg(feature = "migraphx")]
                {
                    let ep = ort::execution_providers::MIGraphXExecutionProvider::default()
                        .with_device_id(id as _)
                        .with_fp16(self.hardware.migraphx.fp16)
                        .with_exhaustive_tune(self.hardware.migraphx.exhaustive_tune);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register MIGraphX: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "MIGraphX")),
                    }
                }
            }
            Device::Vitis => {
                #[cfg(not(feature = "vitis"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "VitisAI")
                        .replace("#FEATURE", "vitis"));
                }
                #[cfg(feature = "vitis")]
                {
                    let ep = ort::execution_providers::VitisAIExecutionProvider::default()
                        .with_cache_dir(
                            crate::Dir::Cache
                                .crate_dir_default_with_subs(&["caches", "vitis"])?
                                .display()
                                .to_string(),
                        );
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register VitisAI: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "VitisAI")),
                    }
                }
            }
            Device::Azure => {
                #[cfg(not(feature = "azure"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "Azure")
                        .replace("#FEATURE", "azure"));
                }
                #[cfg(feature = "azure")]
                {
                    let ep = ort::execution_providers::AzureExecutionProvider::default();
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register Azure: {}", err)
                            })?;
                            builder = builder.with_extensions()?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "Azure")),
                    }
                }
            }
            _ => {
                let ep = ort::execution_providers::CPUExecutionProvider::default()
                    .with_arena_allocator(self.hardware.cpu.arena_allocator);
                match ep.is_available() {
                    Ok(true) => {
                        ep.register(&mut builder)
                            .map_err(|err| anyhow::anyhow!("Failed to register Cpu: {}", err))?;
                    }
                    _ => unreachable!("CPU EP is not available. This case should ideally not be reached under normal circumstances."),
                }
            }
        }

        // session
        let graph_opt_level = match self.graph_opt_level {
            Some(0) => GraphOptimizationLevel::Disable,
            Some(1) => GraphOptimizationLevel::Level1,
            Some(2) => GraphOptimizationLevel::Level2,
            _ => GraphOptimizationLevel::Level3,
        };
        let session = builder
            .with_optimization_level(graph_opt_level)?
            .with_intra_threads(self.num_intra_threads.unwrap_or(n_threads_available))?
            .with_inter_threads(self.num_inter_threads.unwrap_or(2))?
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

    fn get_ort_dtype_from_proto_dtype_id(value: i32) -> Option<TensorElementType> {
        match value {
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
            14 => Some(TensorElementType::Complex64),
            15 => Some(TensorElementType::Complex128),
            16 => Some(TensorElementType::Bfloat16),
            17 => Some(TensorElementType::Float8E4M3FN),
            18 => Some(TensorElementType::Float8E4M3FNUZ),
            19 => Some(TensorElementType::Float8E5M2),
            20 => Some(TensorElementType::Float8E5M2FNUZ),
            21 => Some(TensorElementType::Uint4),
            22 => Some(TensorElementType::Int4),
            _ => None, // 23: Float4e2m1, 0: Undefined
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
            let tensor_type = match Self::get_ort_dtype_from_proto_dtype_id(tensor_type) {
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
        let onnx = self.onnx.as_ref()?;
        match onnx.session.metadata() {
            Err(_) => None,
            Ok(metadata) => metadata.custom(key).ok().flatten(),
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
                .map(|x| DType::from(*x))
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
                .map(|x| DType::from(*x))
                .collect::<Vec<DType>>()
                .into()
        })
    }

    pub fn profile(&self) {
        crate::core::global_ts::global_ts_manager().print_global_summary();
    }

    pub fn info(&self) {
        let info = format!(
            "Minimum Supported Ort Version: 1.{}.x, Opset Version: {}, Device: {}, Parameters: {}, Memory: {}",
            ort::MINOR_VERSION,
            self.opset_version().map_or("Unknown".to_string(), |x| x.to_string()),
            self.device,
            match self.params {
                Some(bytes) if bytes != 0 => {
                    human_bytes_binary(bytes as f64, 2)
                }
                _ => "Unknown".to_string(),
            },
            match self.wbmems {
                Some(bytes) if bytes != 0 => {
                    human_bytes_binary(bytes as f64, 2)
                }
                _ => "Unknown".to_string(),
            },
        );

        info!("{}", info);
    }
}
