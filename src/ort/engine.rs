use aksr::Builder;
use anyhow::Result;
use half::{bf16, f16};
use ndarray::{Array, IxDyn};
use ort::{
    execution_providers::ExecutionProvider,
    session::{builder::GraphOptimizationLevel, input::SessionInputs, Session, SessionInputValue},
    tensor::TensorElementType,
    value::{DynValue, Value},
};
use prost::Message;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tracing::{info, warn};

use crate::{
    human_bytes_binary, onnx, DType, Device, EngineInputs, FromConfig, Iiix, MinOptMax, ORTConfig,
    Ops, XAny, Xs, X,
};

/// ONNX Runtime tensor attributes containing names, data types, dimensions, and index mapping.
#[derive(Debug, Clone, Default)]
pub struct OrtTensorAttr {
    /// Tensor names.
    pub names: Vec<String>,
    /// Mapping from name to its index.
    pub name_to_index: HashMap<String, usize>,
    /// Tensor data types.
    pub dtypes: Vec<TensorElementType>,
    /// Tensor dimensions for each tensor.
    pub dimss: Vec<Vec<usize>>,
    /// Resolved min-opt-max constraints for each dimension.
    pub minoptmax: Vec<Vec<MinOptMax>>,
}

impl OrtTensorAttr {
    pub fn new(
        names: Vec<String>,
        dtypes: Vec<TensorElementType>,
        dimss: Vec<Vec<usize>>,
        minoptmax: Vec<Vec<MinOptMax>>,
    ) -> Self {
        let name_to_index = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        Self {
            names,
            name_to_index,
            dtypes,
            dimss,
            minoptmax,
        }
    }
}

/// Global metadata for an ONNX model.
#[derive(Debug, Clone, Default)]
pub struct ONNXMetadata {
    /// IR version.
    pub ir_version: Option<usize>,
    /// Opset version.
    pub opset_version: Option<usize>,
    /// Producer name.
    pub producer_name: Option<String>,
    /// Producer version.
    pub producer_version: Option<String>,
    /// Model version.
    pub model_version: Option<usize>,
}

/// ONNX Runtime inference engine with core execution state.
#[derive(Debug, Builder, Default)]
pub struct Engine {
    /// Model file path.
    pub file: String,
    /// Model specification string.
    pub spec: String,
    /// Execution device.
    pub device: Device,
    /// Total parameters in the model.
    #[args(alias = "parameters")]
    pub params: Option<usize>,
    /// Estimated weights/biases memory usage.
    #[args(alias = "memory")]
    pub wbmems: Option<usize>,
    /// Model inputs.
    pub inputs: OrtTensorAttr,
    /// Model outputs.
    pub outputs: Arc<OrtTensorAttr>,
    /// ONNX Runtime session.
    pub(crate) session: Option<Session>,
    /// Model metadata.
    pub metadata: ONNXMetadata,
}

impl FromConfig for Engine {
    type Config = ORTConfig;

    fn from_config(config: ORTConfig) -> Result<Self> {
        let span = tracing::info_span!("engine_init", spec = %config.spec);
        let _enter = span.enter();

        let mut params: usize = 0;
        let mut wbmems: usize = 0;
        let metadata: ONNXMetadata;
        let mut input_metadata: OrtTensorAttr;
        let output_metadata: Arc<OrtTensorAttr>;
        let inputs_minoptmax: Vec<Vec<MinOptMax>>;
        let session: Session;

        crate::perf!(&format!("ORT Engine ({})::init", config.spec), {
            let proto = Self::load_onnx(&config.file)?;
            let graph = match &proto.graph {
                Some(graph) => graph,
                None => {
                    anyhow::bail!(
                        "No graph found in this proto. Invalid ONNX model: {}",
                        config.file
                    )
                }
            };

            // params & mems
            let byte_alignment = 16;
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

            // model metadata
            metadata = ONNXMetadata {
                ir_version: Some(proto.ir_version as usize),
                opset_version: proto.opset_import.first().map(|x| x.version as usize),
                producer_name: Some(proto.producer_name.clone()),
                producer_version: Some(proto.producer_version.clone()),
                model_version: Some(proto.model_version as usize),
            };

            // inputs & outputs metadata
            input_metadata = Self::io_from_onnx_value_info(&initializer_names, &graph.input)?;
            output_metadata = Arc::new(Self::io_from_onnx_value_info(
                &initializer_names,
                &graph.output,
            )?);

            inputs_minoptmax = Self::build_ort_inputs(&input_metadata, &config.iiixs)?;

            // session
            // ort::init().commit();
            session = Self::build_session(
                &config.file,
                config.device,
                &config,
                &input_metadata,
                &inputs_minoptmax,
            )?;

            input_metadata.minoptmax = inputs_minoptmax.clone();
        });

        let mut engine = Self {
            file: config.file,
            spec: config.spec,
            device: config.device,
            params: Some(params),
            wbmems: Some(wbmems),
            metadata,
            inputs: input_metadata,
            outputs: output_metadata,
            session: Some(session),
        };

        engine.dry_run(config.num_dry_run, &inputs_minoptmax)?;
        engine.info();

        Ok(engine)
    }
}

impl Engine {
    pub fn dry_run(
        &mut self,
        num_dry_run: usize,
        inputs_minoptmax: &[Vec<MinOptMax>],
    ) -> Result<()> {
        if num_dry_run > 0 {
            // pb
            let pb = crate::PB::dry_run(num_dry_run as u64).with_message(self.spec());

            // dummy inputs
            let mut xs: Vec<X> = Vec::new();
            for i in inputs_minoptmax.iter() {
                let mut shape: Vec<usize> = Vec::new();
                for i_ in i.iter() {
                    shape.push(i_.opt());
                }

                // TODO: directly use ort value
                let x: Array<f32, IxDyn> = Array::ones(shape).into_dyn();
                xs.push(X::from(x));
            }

            // run with alignment (uses internal conversion)
            for i in 0..num_dry_run {
                pb.inc(1);
                crate::perf!(&format!("ORT Engine ({})::dry-run-{}", self.spec, i), {
                    let _ = self.run(&xs)?;
                });
            }

            // update
            pb.finish(Some(&format!(
                "{}({}) on {}",
                self.spec,
                match self.params {
                    Some(bytes) if bytes != 0 => {
                        human_bytes_binary(bytes as f64, 2)
                    }
                    _ => "Unknown".to_string(),
                },
                self.device,
            )));
        }
        Ok(())
    }

    pub fn run<'a, 'i, 'v: 'i, const N: usize>(
        &mut self,
        input_values: impl Into<EngineInputs<'a, 'i, 'v, N>>,
    ) -> Result<Xs<'_>> {
        let span = tracing::debug_span!("engine_run", spec = %self.spec);
        let _enter = span.enter();

        if let Some(session) = &mut self.session {
            let engine_inputs: EngineInputs<'a, 'i, 'v, N> = input_values.into();

            let outputs = match engine_inputs {
                EngineInputs::ProcessedSlice(tensors) => {
                    // XAny path: supports zero-copy CUDA inputs
                    // Note: CUDA outputs will be transferred to CPU in Xs::get_actual()
                    let inputs: Vec<SessionInputValue<'_>> =
                        crate::perf!(&format!("ORT Engine ({})::preprocess", self.spec), {
                            let mut result = Vec::with_capacity(tensors.len());
                            for (tensor, dtype) in tensors.iter().zip(self.inputs.dtypes.iter()) {
                                match tensor {
                                    XAny::Host(x) => {
                                        // CPU path: same as XSlice
                                        if *dtype == TensorElementType::Float32
                                            && x.0.is_standard_layout()
                                        {
                                            let tensor_ref =
                                                ort::value::TensorRef::from_array_view(x.0.view())
                                                    .expect("Failed to create TensorRef");
                                            result.push(SessionInputValue::from(tensor_ref));
                                        } else {
                                            result.push(Self::preprocess(x, dtype)?.into());
                                        }
                                    }
                                    #[cfg(feature = "cuda-runtime")]
                                    XAny::Device(cuda_tensor) => {
                                        // CUDA zero-copy input
                                        result.push(Self::cuda_tensor_to_ort(cuda_tensor, dtype)?);
                                    }
                                }
                            }
                            result
                        });
                    crate::perf!(
                        &format!("ORT Engine ({})::inference", self.spec),
                        session.run(&inputs[..])?
                    )
                }
                EngineInputs::XSlice(xs) => {
                    // &[X] path: try zero-copy when all inputs are f32 and standard layout
                    let all_f32_standard =
                        xs.iter().zip(self.inputs.dtypes.iter()).all(|(x, dtype)| {
                            *dtype == TensorElementType::Float32 && x.0.is_standard_layout()
                        });

                    if all_f32_standard {
                        // Fast path: zero-copy for all inputs
                        let inputs: Vec<SessionInputValue<'_>> =
                            crate::perf!(&format!("ORT Engine ({})::preprocess", self.spec), {
                                xs.iter()
                                    .map(|x| {
                                        let tensor_ref =
                                            ort::value::TensorRef::from_array_view(x.0.view())
                                                .expect("Failed to create TensorRef");
                                        SessionInputValue::from(tensor_ref)
                                    })
                                    .collect()
                            });
                        crate::perf!(
                            &format!("ORT Engine ({})::inference", self.spec),
                            session.run(&inputs[..])?
                        )
                    } else {
                        // Slow path: dtype alignment needed
                        let aligned: Vec<SessionInputValue<'static>> =
                            crate::perf!(&format!("ORT Engine ({})::preprocess", self.spec), {
                                let mut result = Vec::with_capacity(xs.len());
                                for (x, dtype) in xs.iter().zip(self.inputs.dtypes.iter()) {
                                    result.push(Self::preprocess(x, dtype)?.into());
                                }
                                result
                            });
                        crate::perf!(
                            &format!("ORT Engine ({})::inference", self.spec),
                            session.run(&aligned[..])?
                        )
                    }
                }
                EngineInputs::Session(input_values) => match input_values {
                    SessionInputs::ValueArray(input_values) => {
                        // Check if any conversion is needed (fast path optimization)
                        let any_needs_convert = input_values
                            .iter()
                            .zip(self.inputs.dtypes.iter())
                            .any(|(input_value, dtype)| match input_value.dtype() {
                                ort::value::ValueType::Tensor { ty, .. } => *ty != *dtype,
                                _ => false,
                            });

                        if !any_needs_convert {
                            // Fast path: all dtypes match, use array directly without allocation
                            crate::perf!(
                                &format!("ORT Engine ({})::inference", self.spec),
                                session.run(&input_values[..])?
                            )
                        } else {
                            // Slow path: need dtype conversion
                            let aligned: Vec<SessionInputValue<'v>> = crate::perf!(
                                &format!("ORT Engine ({})::preprocess", self.spec),
                                {
                                    let mut xs_ = Vec::with_capacity(input_values.len());
                                    for (input_value, dtype_expected) in
                                        input_values.into_iter().zip(self.inputs.dtypes.iter())
                                    {
                                        let needs_convert = match input_value.dtype() {
                                            ort::value::ValueType::Tensor { ty, .. } => {
                                                *ty != *dtype_expected
                                            }
                                            _ => false,
                                        };

                                        if !needs_convert {
                                            xs_.push(input_value);
                                            continue;
                                        }

                                        // Only convert mismatched inputs. Note: extraction requires CPU-accessible memory.
                                        if let Ok(tensor) = input_value
                                            .downcast_ref::<ort::value::DynTensorValueType>(
                                        ) {
                                            let mem = tensor.memory_info();
                                            if !mem.is_cpu_accessible() {
                                                anyhow::bail!(
                                                    "Cannot dtype-convert a non-CPU-accessible input (device={:?}, device_id={}). Consider providing this input on CPU or matching the model's expected dtype.",
                                                    mem.allocation_device(),
                                                    mem.device_id()
                                                );
                                            }
                                        }

                                        let view = input_value.try_extract_array::<f32>()?;
                                        xs_.push(
                                            Self::preprocess_view(&view, dtype_expected)?.into(),
                                        );
                                    }
                                    xs_
                                }
                            );

                            crate::perf!(
                                &format!("ORT Engine ({})::inference", self.spec),
                                session.run(&aligned[..])?
                            )
                        }
                    }
                    SessionInputs::ValueSlice(input_values) => {
                        // Check if any conversion is needed
                        let any_needs_convert = input_values
                            .iter()
                            .zip(self.inputs.dtypes.iter())
                            .any(|(input_value, dtype)| {
                                matches!(input_value.dtype(),
                                    ort::value::ValueType::Tensor { ty, .. } if *ty != *dtype)
                            });

                        if !any_needs_convert {
                            // Fast path: all dtypes match
                            crate::perf!(
                                &format!("ORT Engine ({})::inference", self.spec),
                                session.run(input_values)?
                            )
                        } else {
                            // Slow path: need dtype conversion
                            let aligned: Vec<SessionInputValue<'i>> = crate::perf!(
                                &format!("ORT Engine ({})::preprocess", self.spec),
                                {
                                    let mut xs_ = Vec::with_capacity(input_values.len());
                                    for (input_value, dtype_expected) in
                                        input_values.iter().zip(self.inputs.dtypes.iter())
                                    {
                                        let needs_convert = matches!(
                                            input_value.dtype(),
                                            ort::value::ValueType::Tensor { ty, .. } if *ty != *dtype_expected
                                        );

                                        if !needs_convert {
                                            // Re-borrow without cloning underlying buffers.
                                            xs_.push(SessionInputValue::from(&**input_value));
                                            continue;
                                        }

                                        if let Ok(tensor) = input_value
                                            .downcast_ref::<ort::value::DynTensorValueType>(
                                        ) {
                                            let mem = tensor.memory_info();
                                            if !mem.is_cpu_accessible() {
                                                anyhow::bail!(
                                                    "Cannot dtype-convert a non-CPU-accessible input (device={:?}, device_id={}). Consider providing this input on CPU or matching the model's expected dtype.",
                                                    mem.allocation_device(),
                                                    mem.device_id()
                                                );
                                            }
                                        }

                                        let view = input_value.try_extract_array::<f32>()?;
                                        xs_.push(
                                            Self::preprocess_view(&view, dtype_expected)?.into(),
                                        );
                                    }
                                    xs_
                                }
                            );

                            crate::perf!(
                                &format!("ORT Engine ({})::inference", self.spec),
                                session.run(&aligned[..])?
                            )
                        }
                    }
                    input_values => crate::perf!(
                        &format!("ORT Engine ({})::inference", self.spec),
                        session.run(input_values)?
                    ),
                },
            };

            // Use shared output metadata to avoid redundant allocations
            Ok(Xs::with_metadata(outputs, Arc::clone(&self.outputs)))
        } else {
            anyhow::bail!("Failed to run with ONNXRuntime. No model info found.");
        }
    }

    /// Convert CUDA tensor to ORT input (zero-copy).
    #[cfg(feature = "cuda-runtime")]
    fn cuda_tensor_to_ort<'a>(
        cuda_tensor: &'a crate::XCuda,
        dtype: &TensorElementType,
    ) -> Result<SessionInputValue<'a>> {
        use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
        use ort::tensor::Shape;
        use ort::value::TensorRefMut;

        // Only f32 is supported for now (can extend later)
        if *dtype != TensorElementType::Float32 {
            anyhow::bail!(
                "CUDA zero-copy only supports Float32, got {dtype:?}. Consider using CPU preprocessing."
            );
        }

        // Create MemoryInfo for CUDA device
        let mem_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            cuda_tensor.device_id() as i32,
            AllocatorType::Device,
            MemoryType::Default,
        )?;

        // Create TensorRefMut from raw CUDA pointer (f32 type)
        let tensor_ref: TensorRefMut<'a, _> = unsafe {
            TensorRefMut::<f32>::from_raw(
                mem_info,
                cuda_tensor.device_ptr() as *mut std::ffi::c_void,
                Shape::from(cuda_tensor.shape_i64()),
            )?
        };

        Ok(SessionInputValue::from(tensor_ref))
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

    /// Preprocessing from ArrayView (avoids intermediate copy).
    fn preprocess_view(
        view: &ndarray::ArrayViewD<'_, f32>,
        dtype: &TensorElementType,
    ) -> Result<DynValue> {
        let x = match dtype {
            TensorElementType::Float32 => Value::from_array(view.to_owned())?.into_dyn(),
            TensorElementType::Float64 => Value::from_array(view.mapv(|x_| x_ as f64))?.into_dyn(),
            TensorElementType::Float16 => Value::from_array(view.mapv(f16::from_f32))?.into_dyn(),
            TensorElementType::Bfloat16 => Value::from_array(view.mapv(bf16::from_f32))?.into_dyn(),
            TensorElementType::Int8 => Value::from_array(view.mapv(|x_| x_ as i8))?.into_dyn(),
            TensorElementType::Int16 => Value::from_array(view.mapv(|x_| x_ as i16))?.into_dyn(),
            TensorElementType::Int32 => Value::from_array(view.mapv(|x_| x_ as i32))?.into_dyn(),
            TensorElementType::Int64 => Value::from_array(view.mapv(|x_| x_ as i64))?.into_dyn(),
            TensorElementType::Uint8 => Value::from_array(view.mapv(|x_| x_ as u8))?.into_dyn(),
            TensorElementType::Uint16 => Value::from_array(view.mapv(|x_| x_ as u16))?.into_dyn(),
            TensorElementType::Uint32 => Value::from_array(view.mapv(|x_| x_ as u32))?.into_dyn(),
            TensorElementType::Uint64 => Value::from_array(view.mapv(|x_| x_ as u64))?.into_dyn(),
            TensorElementType::Bool => Value::from_array(view.mapv(|x_| x_ != 0.))?.into_dyn(),
            _ => unimplemented!(),
        };
        Ok(x)
    }

    #[allow(unused_variables)]
    fn build_session(
        model_file: &str,
        device: Device,
        config: &ORTConfig,
        inputs: &OrtTensorAttr,
        inputs_minoptmax: &[Vec<MinOptMax>],
    ) -> Result<Session> {
        #[allow(unused_mut)]
        let mut builder = Session::builder()?;
        let compile_help = "Please compile ONNXRuntime with #EP";
        let feature_help = "#EP EP requires the features: `#FEATURE`. \
            \nConsider enabling them by passing, e.g., `--features #FEATURE`";
        let n_threads_available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        match device {
            Device::NvRtx(id) => {
                #[cfg(not(feature = "nvrtx"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "NvTensorRT-RTX")
                        .replace("#FEATURE", "nvrtx"));
                }
                #[cfg(feature = "nvrtx")]
                {
                    let (spec_min, spec_opt, spec_max) =
                        Self::generate_shape_specs(&inputs.names, &inputs_minoptmax)?;

                    let ep = ort::ep::nvrtx::NVRTX::default()
                        .with_device_id(id as _)
                        .with_runtime_cache_path(
                            crate::Dir::Cache
                                .crate_dir_default_with_subs(&["caches", "tensorrt-rtx"])?
                                .display(),
                        )
                        .with_profile_min_shapes(spec_min)
                        .with_profile_opt_shapes(spec_opt)
                        .with_profile_max_shapes(spec_max);

                    match ep.is_available() {
                        Ok(true) => {
                            info!(
                                "Initial model serialization with NVTensorRT-RTX may require a wait..."
                            );
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register NVTensorRT-RTX: {}", err)
                            })?;
                        }
                        _ => {
                            anyhow::bail!(compile_help.replace("#EP", "NVTensorRT-RTX"))
                        }
                    }
                }
            }
            Device::TensorRt(id) => {
                #[cfg(not(feature = "tensorrt"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "TensorRT")
                        .replace("#FEATURE", "tensorrt"));
                }
                #[cfg(feature = "tensorrt")]
                {
                    let (spec_min, spec_opt, spec_max) =
                        Self::generate_shape_specs(&inputs.names, inputs_minoptmax)?;
                    let cache_path =
                        crate::Dir::Cache.crate_dir_default_with_subs(&["caches", "tensorrt"])?;

                    let mut ep = ort::execution_providers::TensorRTExecutionProvider::default()
                        .with_device_id(id as i32)
                        .with_max_workspace_size(config.ep.tensorrt.max_workspace_size)
                        .with_builder_optimization_level(
                            config.ep.tensorrt.builder_optimization_level,
                        )
                        .with_dla_core(config.ep.tensorrt.dla_core)
                        .with_dla(config.ep.tensorrt.dla)
                        .with_detailed_build_log(config.ep.tensorrt.detailed_build_log)
                        .with_fp16(config.ep.tensorrt.fp16)
                        .with_int8(config.ep.tensorrt.int8)
                        .with_int8_use_native_calibration_table(
                            config.ep.tensorrt.int8_use_native_calibration_table,
                        )
                        .with_engine_cache(config.ep.tensorrt.engine_cache)
                        .with_timing_cache(config.ep.tensorrt.timing_cache)
                        .with_dump_ep_context_model(config.ep.tensorrt.dump_ep_context_model)
                        .with_dump_subgraphs(config.ep.tensorrt.dump_subgraphs)
                        .with_min_subgraph_size(config.ep.tensorrt.min_subgraph_size)
                        .with_engine_cache_path(cache_path.display())
                        .with_timing_cache_path(cache_path.display())
                        .with_ep_context_file_path(cache_path.display())
                        .with_profile_min_shapes(spec_min)
                        .with_profile_opt_shapes(spec_opt)
                        .with_profile_max_shapes(spec_max);
                    if config.ep.tensorrt.int8
                        && !config.ep.tensorrt.int8_calibration_table_name.is_empty()
                    {
                        ep = ep.with_int8_calibration_table_name(
                            config.ep.tensorrt.int8_calibration_table_name.clone(),
                        );
                    }

                    match ep.is_available() {
                        Ok(true) => {
                            info!(
                                "Initial model serialization with TensorRT may require a wait..."
                            );
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register TensorRT: {err}")
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
                        .with_device_id(id as i32)
                        .with_conv_max_workspace(config.ep.cuda.conv_max_workspace)
                        .with_prefer_nhwc(config.ep.cuda.prefer_nhwc)
                        .with_tf32(config.ep.cuda.tf32)
                        .with_fuse_conv_bias(config.ep.cuda.fuse_conv_bias)
                        .with_cuda_graph(config.ep.cuda.cuda_graph);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder)
                                .map_err(|err| anyhow::anyhow!("Failed to register CUDA: {err}"))?;
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
                    use ort::execution_providers::coreml::{
                        ComputeUnits, ModelFormat, SpecializationStrategy,
                    };

                    let ep = ort::execution_providers::CoreMLExecutionProvider::default()
                        .with_model_cache_dir(
                            crate::Dir::Cache
                                .crate_dir_default_with_subs(&["caches", "coreml"])?
                                .display(),
                        )
                        .with_static_input_shapes(config.ep.coreml.static_input_shapes)
                        .with_subgraphs(config.ep.coreml.subgraph_running)
                        .with_compute_units(match config.ep.coreml.compute_units {
                            0 => ComputeUnits::All,
                            1 => ComputeUnits::CPUAndGPU,
                            2 => ComputeUnits::CPUAndNeuralEngine,
                            3 => ComputeUnits::CPUOnly,
                            _ => ComputeUnits::All,
                        })
                        .with_model_format(match config.ep.coreml.model_format {
                            0 => ModelFormat::MLProgram,
                            1 => ModelFormat::NeuralNetwork,
                            _ => ModelFormat::MLProgram,
                        })
                        .with_specialization_strategy(
                            match config.ep.coreml.specialization_strategy {
                                1 => SpecializationStrategy::FastPrediction,
                                _ => SpecializationStrategy::Default,
                            },
                        );

                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register CoreML: {err}")
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "CoreML")),
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
                                anyhow::anyhow!("Failed to register DirectML: {err}")
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "DirectML")),
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
                        .with_num_threads(config.ep.openvino.num_threads)
                        .with_dynamic_shapes(config.ep.openvino.dynamic_shapes)
                        .with_opencl_throttling(config.ep.openvino.opencl_throttling)
                        .with_qdq_optimizer(config.ep.openvino.qdq_optimizer)
                        .with_cache_dir(
                            crate::Dir::Cache
                                .crate_dir_default_with_subs(&["caches", "openvino"])?
                                .display()
                                .to_string(),
                        );
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register OpenVINO: {err}")
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "OpenVINO")),
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
                        .with_arena_allocator(config.ep.onednn.arena_allocator);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register oneDNN: {err}")
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "oneDNN")),
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
                        .with_cann_graph(config.ep.cann.graph_inference)
                        .with_dump_graphs(config.ep.cann.dump_graphs)
                        .with_dump_om_model(config.ep.cann.dump_om_model);

                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder)
                                .map_err(|err| anyhow::anyhow!("Failed to register CANN: {err}"))?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "CANN")),
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
                        .with_device_id(id as i32);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder)
                                .map_err(|err| anyhow::anyhow!("Failed to register QNN: {err}"))?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "QNN")),
                    }
                }
            }
            Device::MiGraphX(id) => {
                #[cfg(not(feature = "migraphx"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "MiGraphX")
                        .replace("#FEATURE", "migraphx"));
                }
                #[cfg(feature = "migraphx")]
                {
                    let ep = ort::execution_providers::MIGraphXExecutionProvider::default()
                        .with_device_id(id as i32)
                        .with_fp16(config.ep.migraphx.fp16)
                        .with_exhaustive_tune(config.ep.migraphx.exhaustive_tune);

                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register MiGraphX: {err}")
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "MiGraphX")),
                    }
                }
            }
            Device::Xnnpack => {
                #[cfg(not(feature = "xnnpack"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "XNNPACK")
                        .replace("#FEATURE", "xnnpack"));
                }
                #[cfg(feature = "xnnpack")]
                {
                    let ep = ort::execution_providers::XNNPACKExecutionProvider::default();
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register XNNPACK: {err}")
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "XNNPACK")),
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
                                anyhow::anyhow!("Failed to register RKNPU: {err}")
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "RKNPU")),
                    }
                }
            }
            Device::Acl => {
                #[cfg(not(feature = "acl"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "ACL")
                        .replace("#FEATURE", "acl"));
                }
                #[cfg(feature = "acl")]
                {
                    let ep = ort::execution_providers::ACLExecutionProvider::default()
                        .with_fast_math(true);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder)
                                .map_err(|err| anyhow::anyhow!("Failed to register ACL: {err}"))?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "ACL")),
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
                        .with_cpu_only(config.ep.nnapi.cpu_only)
                        .with_disable_cpu(config.ep.nnapi.disable_cpu)
                        .with_fp16(config.ep.nnapi.fp16)
                        .with_nchw(config.ep.nnapi.nchw);

                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register NNAPI: {err}")
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
                        .with_arena_allocator(config.ep.armnn.arena_allocator);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register ArmNN: {err}")
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "ArmNN")),
                    }
                }
            }
            Device::Vitis => {
                #[cfg(not(feature = "vitis"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "Vitis")
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
                                anyhow::anyhow!("Failed to register Vitis: {err}")
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "VitisAI")),
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
                            ep.register(&mut builder)
                                .map_err(|err| anyhow::anyhow!("Failed to register TVM: {err}"))?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "TVM")),
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
                                anyhow::anyhow!("Failed to register Azure: {err}")
                            })?;
                            builder = builder.with_extensions()?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "Azure")),
                    }
                }
            }
            Device::Wgpu(_) => {
                #[cfg(not(feature = "webgpu"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "WebGPU")
                        .replace("#FEATURE", "webgpu"));
                }
                #[cfg(feature = "webgpu")]
                {
                    let ep = ort::execution_providers::WebGPUExecutionProvider::default();
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register WebGPU: {err}")
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "WebGPU")),
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
                        .with_device_id(id as i32);
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder)
                                .map_err(|err| anyhow::anyhow!("Failed to register ROCm: {err}"))?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "ROCm")),
                    }
                }
            }
            Device::Cpu(_) => {
                let ep = ort::execution_providers::CPUExecutionProvider::default()
                    .with_arena_allocator(config.ep.cpu.arena_allocator);
                match ep.is_available() {
                    Ok(true) => {
                        ep.register(&mut builder)
                            .map_err(|err| anyhow::anyhow!("Failed to register CPU: {err}"))?;
                    }
                    _ => anyhow::bail!(compile_help.replace("#EP", "CPU")),
                }
            }
        }

        // threads
        builder =
            builder.with_intra_threads(config.num_intra_threads.unwrap_or(n_threads_available))?;
        builder = builder.with_inter_threads(config.num_inter_threads.unwrap_or(8))?;

        // optimization
        #[cfg(not(feature = "tensorrt"))]
        if let Some(level) = config.graph_opt_level {
            builder = builder.with_optimization_level(match level {
                0 => GraphOptimizationLevel::Disable,
                1 => GraphOptimizationLevel::Level1,
                2 => GraphOptimizationLevel::Level2,
                3 => GraphOptimizationLevel::Level3,
                _ => anyhow::bail!("Invalid graph optimization level: {level}"),
            })?;
        }
        #[cfg(feature = "tensorrt")]
        {
            tracing::info!("Disabling ort graph optimization for TensorRT. `ort_graph_opt_level` setting is ignored.");
            builder = builder.with_optimization_level(GraphOptimizationLevel::Disable)?;
        }

        let session = builder.commit_from_file(model_file)?;
        Ok(session)
    }

    /// Generate shape specifications for TensorRT and TensorRT-RTX
    #[inline]
    #[allow(unused)]
    fn generate_shape_specs(
        names: &[String],
        inputs_minoptmax: &[Vec<crate::MinOptMax>],
    ) -> anyhow::Result<(String, String, String)> {
        anyhow::ensure!(
            names.len() == inputs_minoptmax.len(),
            "Failed to generate shape specs: names and inputs_minoptmax length mismatch"
        );

        use std::fmt::Write;

        let n = 128;
        let mut spec_min = String::with_capacity(n);
        let mut spec_opt = String::with_capacity(n);
        let mut spec_max = String::with_capacity(n);

        for (i, name) in names.iter().enumerate() {
            if i != 0 {
                spec_min.push(',');
                spec_opt.push(',');
                spec_max.push(',');
            }

            // Write name prefix
            spec_min.push_str(name);
            spec_min.push(':');
            spec_opt.push_str(name);
            spec_opt.push(':');
            spec_max.push_str(name);
            spec_max.push(':');

            // Write dimensions
            for (j, d) in inputs_minoptmax[i].iter().enumerate() {
                if j != 0 {
                    spec_min.push('x');
                    spec_opt.push('x');
                    spec_max.push('x');
                }

                // Use write! for more efficient number formatting
                write!(spec_min, "{}", d.min())
                    .map_err(|_| anyhow::anyhow!("Failed to write min shape for input {name}"))?;
                write!(spec_opt, "{}", d.opt())
                    .map_err(|_| anyhow::anyhow!("Failed to write opt shape for input {name}"))?;
                write!(spec_max, "{}", d.max())
                    .map_err(|_| anyhow::anyhow!("Failed to write max shape for input {name}"))?;
            }
        }

        Ok((spec_min, spec_opt, spec_max))
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
        Ok(OrtTensorAttr::new(names, dtypes, dimss, vec![]))
    }

    pub fn load_onnx<P: AsRef<std::path::Path>>(p: P) -> Result<onnx::ModelProto> {
        let path_ref = p.as_ref();
        let f = std::fs::read(path_ref).map_err(|err| {
            anyhow::anyhow!("Failed to read ONNX file '{path_ref:?}': {err}. Error: {err}")
        })?;
        onnx::ModelProto::decode(f.as_slice()).map_err(|err| {
            anyhow::anyhow!(
                "Failed to read the ONNX model: The file might be incomplete or corrupted. More detailed: {err}"
            )
        })
    }

    pub fn batch(&self) -> &MinOptMax {
        &self.inputs.minoptmax[0][0]
    }

    pub fn is_batch_dyn(&self) -> bool {
        self.batch().is_dyn()
    }

    pub fn try_height(&self) -> Option<&MinOptMax> {
        self.inputs.minoptmax.first().and_then(|x| x.get(2))
    }

    pub fn height(&self) -> &MinOptMax {
        // unsafe
        &self.inputs.minoptmax[0][2]
    }

    pub fn is_height_dyn(&self) -> bool {
        self.height().is_dyn()
    }

    pub fn try_width(&self) -> Option<&MinOptMax> {
        self.inputs.minoptmax.first().and_then(|x| x.get(3))
    }

    pub fn width(&self) -> &MinOptMax {
        // unsafe
        &self.inputs.minoptmax[0][3]
    }

    pub fn is_width_dyn(&self) -> bool {
        self.width().is_dyn()
    }

    pub fn try_fetch(&self, key: &str) -> Option<String> {
        let session = self.session.as_ref()?;
        match session.metadata() {
            Err(_) => None,
            Ok(metadata) => metadata.custom(key),
        }
    }

    // TODO: use ort.session.metadata()
    pub fn ir_version(&self) -> Option<usize> {
        self.metadata.ir_version
    }

    // TODO: use ort.session.metadata()
    pub fn opset_version(&self) -> Option<usize> {
        self.metadata.opset_version
    }

    // TODO: use ort.session.metadata()
    pub fn producer_name(&self) -> Option<String> {
        self.metadata.producer_name.clone()
    }

    // TODO: use ort.session.metadata()
    pub fn producer_version(&self) -> Option<String> {
        self.metadata.producer_version.clone()
    }

    // TODO: use ort.session.metadata()
    pub fn model_version(&self) -> Option<usize> {
        self.metadata.model_version
    }

    pub fn ishapes(&self) -> Option<&[Vec<usize>]> {
        if self.session.is_some() {
            Some(self.inputs.dimss.as_slice())
        } else {
            None
        }
    }

    pub fn idimss(&self) -> Option<&[Vec<usize>]> {
        if self.session.is_some() {
            Some(self.inputs.dimss.as_slice())
        } else {
            None
        }
    }

    pub fn inames(&self) -> Option<&[String]> {
        if self.session.is_some() {
            Some(self.inputs.names.as_slice())
        } else {
            None
        }
    }

    pub fn idtypes(&self) -> Option<Vec<DType>> {
        if self.session.is_some() {
            self.inputs
                .dtypes
                .iter()
                .map(|x| DType::from(*x))
                .collect::<Vec<DType>>()
                .into()
        } else {
            None
        }
    }

    pub fn oshapes(&self) -> Option<&[Vec<usize>]> {
        if self.session.is_some() {
            Some(self.outputs.dimss.as_slice())
        } else {
            None
        }
    }

    pub fn odimss(&self) -> Option<&[Vec<usize>]> {
        if self.session.is_some() {
            Some(self.outputs.dimss.as_slice())
        } else {
            None
        }
    }

    pub fn onames(&self) -> Option<&[String]> {
        if self.session.is_some() {
            Some(self.outputs.names.as_slice())
        } else {
            None
        }
    }

    pub fn odtypes(&self) -> Option<Vec<DType>> {
        if self.session.is_some() {
            self.outputs
                .dtypes
                .iter()
                .map(|x| DType::from(*x))
                .collect::<Vec<DType>>()
                .into()
        } else {
            None
        }
    }

    pub fn profile(&self) {
        crate::perf_chart();
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
