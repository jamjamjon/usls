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
use std::collections::HashSet;
use tracing::{info, warn};

#[cfg(feature = "cuda")]
use crate::XCuda;
use crate::{
    build_progress_bar, elapsed_global, human_bytes_binary, onnx, DType, Device, EpConfig,
    FromConfig, Iiix, MinOptMax, ORTConfig, Ops, XAny, Xs, PROGRESS_BAR_STYLE_CYAN_2,
    PROGRESS_BAR_STYLE_FINISH, X,
};

/// Pre-converted ORT input value (owned), intended for dynamic inputs.
///
/// This is a `usls`-level alias so call sites do not need to import `ort` types.
pub type OrtInput = SessionInputValue<'static>;

/// A list of pre-converted ORT input values (owned).
pub type OrtInputs = Vec<OrtInput>;

/// Unified input type for `Engine::run()`.
///
/// Supports multiple input formats with automatic dtype alignment:
/// - `ort_inputs![...]` macro output
/// - `OrtInputs` / `Vec<SessionInputValue>`
/// - `&[X]` / `&Vec<X>` slices
/// - `&[XAny]` / `&XAny` (supports zero-copy CUDA)
pub enum EngineInputs<'a, 'i, 'v, const N: usize> {
    /// Pre-converted ORT inputs (from ort_inputs! or manual conversion)
    Session(SessionInputs<'i, 'v, N>),
    /// Raw X slices that need dtype alignment
    XSlice(&'a [X]),
    /// XAny slice (supports CUDA zero-copy)
    ProcessedSlice(&'a [XAny]),
}

impl<'a, 'i, 'v, const N: usize> From<SessionInputs<'i, 'v, N>> for EngineInputs<'a, 'i, 'v, N> {
    fn from(inputs: SessionInputs<'i, 'v, N>) -> Self {
        EngineInputs::Session(inputs)
    }
}

impl<'a> From<&'a [X]> for EngineInputs<'a, 'static, 'static, 0> {
    fn from(xs: &'a [X]) -> Self {
        EngineInputs::XSlice(xs)
    }
}

impl<'a> From<&'a Vec<X>> for EngineInputs<'a, 'static, 'static, 0> {
    fn from(xs: &'a Vec<X>) -> Self {
        EngineInputs::XSlice(xs.as_slice())
    }
}

// Allow fixed-size arrays from ort_inputs! macro
impl<'a, 'v, const N: usize> From<[SessionInputValue<'v>; N]> for EngineInputs<'a, 'v, 'v, N> {
    fn from(arr: [SessionInputValue<'v>; N]) -> Self {
        EngineInputs::Session(SessionInputs::ValueArray(arr))
    }
}

// Allow OrtInputs (Vec<SessionInputValue>)
impl<'a, const N: usize> From<OrtInputs> for EngineInputs<'a, 'static, 'static, N> {
    fn from(inputs: OrtInputs) -> Self {
        EngineInputs::Session(SessionInputs::ValueMap(
            inputs
                .into_iter()
                .enumerate()
                .map(|(i, v)| (std::borrow::Cow::Owned(i.to_string()), v))
                .collect(),
        ))
    }
}

// Allow &[SessionInputValue] slice
impl<'a, 'i, 'v, const N: usize> From<&'i [SessionInputValue<'v>]> for EngineInputs<'a, 'i, 'v, N> {
    fn from(slice: &'i [SessionInputValue<'v>]) -> Self {
        EngineInputs::Session(SessionInputs::ValueSlice(slice))
    }
}

// Allow XAny inputs
impl<'a> From<&'a [XAny]> for EngineInputs<'a, 'static, 'static, 0> {
    fn from(tensors: &'a [XAny]) -> Self {
        EngineInputs::ProcessedSlice(tensors)
    }
}

impl<'a> From<&'a XAny> for EngineInputs<'a, 'static, 'static, 0> {
    fn from(tensor: &'a XAny) -> Self {
        EngineInputs::ProcessedSlice(std::slice::from_ref(tensor))
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
    #[args(extend)]
    pub iiixs: Vec<Iiix>,
    #[args(alias = "parameters")]
    pub params: Option<usize>,
    #[args(alias = "memory")]
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
    pub ep: EpConfig,
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
            graph_opt_level: None,
            num_intra_threads: None,
            num_inter_threads: None,
            ep: EpConfig::new(),
        }
    }
}

impl FromConfig for Engine {
    type Config = ORTConfig;

    fn from_config(config: ORTConfig) -> Result<Self> {
        Self {
            file: config.file,
            spec: config.spec,
            iiixs: config.iiixs,
            device: config.device,
            num_dry_run: config.num_dry_run,
            graph_opt_level: config.graph_opt_level,
            num_intra_threads: config.num_intra_threads,
            num_inter_threads: config.num_inter_threads,
            ep: config.ep,
            ..Default::default()
        }
        .build()
    }
}

impl Engine {
    pub fn build(mut self) -> Result<Self> {
        let span = tracing::info_span!("engine_build", spec = %self.spec);
        let _enter = span.enter();

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

            // dummy inputs
            let mut xs: Vec<X> = Vec::new();
            for i in self.inputs_minoptmax().iter() {
                let mut shape: Vec<usize> = Vec::new();
                for i_ in i.iter() {
                    shape.push(i_.opt());
                }
                let x: Array<f32, IxDyn> = Array::ones(shape).into_dyn();
                xs.push(X::from(x));
            }

            // run with alignment (uses internal conversion)
            for i in 0..self.num_dry_run {
                pb.inc(1);
                let name = format!("[{}] ort_dry_run_{}", self.spec, i);
                elapsed_global!(&name, {
                    let _ = self.run(&xs)?;
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

    pub fn run<'a, 'i, 'v: 'i, const N: usize>(
        &mut self,
        input_values: impl Into<EngineInputs<'a, 'i, 'v, N>>,
    ) -> Result<Xs<'_>> {
        let span = tracing::debug_span!("engine_run", spec = %self.spec);
        let _enter = span.enter();

        if let Some(onnx) = &mut self.onnx {
            let engine_inputs: EngineInputs<'a, 'i, 'v, N> = input_values.into();

            let outputs = match engine_inputs {
                EngineInputs::ProcessedSlice(tensors) => {
                    // XAny path: supports zero-copy CUDA
                    let inputs: Vec<SessionInputValue<'_>> =
                        elapsed_global!(&format!("[{}] ort_preprocessing", self.spec), {
                            let mut result = Vec::with_capacity(tensors.len());
                            for (tensor, dtype) in tensors.iter().zip(onnx.inputs.dtypes.iter()) {
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
                                    #[cfg(feature = "cuda")]
                                    XAny::Device(cuda_tensor) => {
                                        // CUDA zero-copy path
                                        result.push(Self::cuda_tensor_to_ort(cuda_tensor, dtype)?);
                                    }
                                }
                            }
                            result
                        });
                    elapsed_global!(
                        &format!("[{}] ort_inference (cuda-zero-copy)", self.spec),
                        onnx.session.run(&inputs[..])?
                    )
                }
                EngineInputs::XSlice(xs) => {
                    // &[X] path: try zero-copy when all inputs are f32 and standard layout
                    let all_f32_standard =
                        xs.iter().zip(onnx.inputs.dtypes.iter()).all(|(x, dtype)| {
                            *dtype == TensorElementType::Float32 && x.0.is_standard_layout()
                        });

                    if all_f32_standard {
                        // Fast path: zero-copy for all inputs
                        let inputs: Vec<SessionInputValue<'_>> =
                            elapsed_global!(&format!("[{}] ort_preprocessing", self.spec), {
                                xs.iter()
                                    .map(|x| {
                                        let tensor_ref =
                                            ort::value::TensorRef::from_array_view(x.0.view())
                                                .expect("Failed to create TensorRef");
                                        SessionInputValue::from(tensor_ref)
                                    })
                                    .collect()
                            });
                        elapsed_global!(
                            &format!("[{}] ort_inference (zero-copy)", self.spec),
                            onnx.session.run(&inputs[..])?
                        )
                    } else {
                        // Slow path: dtype alignment needed
                        let aligned: Vec<SessionInputValue<'static>> =
                            elapsed_global!(&format!("[{}] ort_preprocessing", self.spec), {
                                let mut result = Vec::with_capacity(xs.len());
                                for (x, dtype) in xs.iter().zip(onnx.inputs.dtypes.iter()) {
                                    result.push(Self::preprocess(x, dtype)?.into());
                                }
                                result
                            });
                        elapsed_global!(
                            &format!("[{}] ort_inference", self.spec),
                            onnx.session.run(&aligned[..])?
                        )
                    }
                }
                EngineInputs::Session(input_values) => match input_values {
                    SessionInputs::ValueArray(input_values) => {
                        let aligned =
                            elapsed_global!(&format!("[{}] ort_preprocessing", self.spec), {
                                let mut xs_ = Vec::with_capacity(input_values.len());
                                for (input_value, dtype) in
                                    input_values.into_iter().zip(onnx.inputs.dtypes.iter())
                                {
                                    // Fast path: if dtype matches, keep as-is
                                    let needs_convert = match input_value.dtype() {
                                        ort::value::ValueType::Tensor { ty, .. } => *ty != *dtype,
                                        _ => false,
                                    };

                                    if !needs_convert {
                                        xs_.push(input_value);
                                        continue;
                                    }

                                    // Extract view and convert dtype
                                    let view = input_value.try_extract_array::<f32>()?;
                                    xs_.push(Self::preprocess_view(&view, dtype)?.into());
                                }
                                xs_
                            });

                        elapsed_global!(
                            &format!("[{}] ort_inference", self.spec),
                            onnx.session.run(&aligned[..])?
                        )
                    }
                    SessionInputs::ValueSlice(input_values) => {
                        // Check if any conversion is needed
                        let any_needs_convert = input_values
                            .iter()
                            .zip(onnx.inputs.dtypes.iter())
                            .any(|(input_value, dtype)| {
                                matches!(input_value.dtype(),
                                    ort::value::ValueType::Tensor { ty, .. } if *ty != *dtype)
                            });

                        if !any_needs_convert {
                            // Fast path: all dtypes match
                            elapsed_global!(
                                &format!("[{}] ort_inference", self.spec),
                                onnx.session.run(input_values)?
                            )
                        } else {
                            // Slow path: need dtype conversion
                            let aligned =
                                elapsed_global!(&format!("[{}] ort_preprocessing", self.spec), {
                                    let mut xs_ = Vec::with_capacity(input_values.len());
                                    for (input_value, dtype) in
                                        input_values.iter().zip(onnx.inputs.dtypes.iter())
                                    {
                                        let view = input_value.try_extract_array::<f32>()?;
                                        xs_.push(Self::preprocess_view(&view, dtype)?.into());
                                    }
                                    xs_
                                });

                            elapsed_global!(
                                &format!("[{}] ort_inference", self.spec),
                                onnx.session.run(&aligned[..])?
                            )
                        }
                    }
                    input_values => elapsed_global!(
                        &format!("[{}] ort_inference", self.spec),
                        onnx.session.run(input_values)?
                    ),
                },
            };

            Ok(Xs::from(outputs))
        } else {
            anyhow::bail!("Failed to run with ONNXRuntime. No model info found.");
        }
    }

    /// Convert CUDA tensor to ORT input (zero-copy).
    #[cfg(feature = "cuda")]
    fn cuda_tensor_to_ort<'a>(
        cuda_tensor: &'a XCuda,
        dtype: &TensorElementType,
    ) -> Result<SessionInputValue<'a>> {
        use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
        use ort::tensor::Shape;
        use ort::value::TensorRefMut;

        // Only f32 is supported for now (can extend later)
        if *dtype != TensorElementType::Float32 {
            anyhow::bail!(
                "CUDA zero-copy only supports Float32, got {:?}. Consider using CPU preprocessing.",
                dtype
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

    /// Optimized preprocessing from ArrayView (avoids intermediate copy).
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

    // fn postprocess(x: &DynValue, dtype: &TensorElementType) -> Result<Array<f32, IxDyn>> {
    //     fn _extract_and_convert<T>(x: &DynValue, map_fn: impl Fn(T) -> f32) -> Array<f32, IxDyn>
    //     where
    //         T: Clone + 'static + ort::tensor::PrimitiveTensorElementType,
    //     {
    //         match x.try_extract_array::<T>() {
    //             Err(err) => {
    //                 debug!("Failed to extract from ort outputs: {:?}. A default value has been generated.", err);
    //                 Array::zeros(0).into_dyn()
    //             }
    //             Ok(x) => x.view().mapv(map_fn).into_owned(),
    //         }
    //     }

    //     // Handle f16/bf16 with unaligned read since ORT's buffer may not be 2-byte aligned
    //     fn _extract_f16_unaligned(x: &DynValue) -> Array<f32, IxDyn> {
    //         let (shape, num_elements, data_ptr) = match _get_tensor_ptr(x) {
    //             Some(v) => v,
    //             None => return Array::zeros(0).into_dyn(),
    //         };
    //         let ptr = data_ptr as *const f16;
    //         let converted: Vec<f32> = (0..num_elements)
    //             .map(|i| unsafe { std::ptr::read_unaligned(ptr.add(i)) }.to_f32())
    //             .collect();
    //         Array::from_shape_vec(IxDyn(&shape), converted)
    //             .unwrap_or_else(|_| Array::zeros(0).into_dyn())
    //     }

    //     fn _extract_bf16_unaligned(x: &DynValue) -> Array<f32, IxDyn> {
    //         let (shape, num_elements, data_ptr) = match _get_tensor_ptr(x) {
    //             Some(v) => v,
    //             None => return Array::zeros(0).into_dyn(),
    //         };
    //         let ptr = data_ptr as *const bf16;
    //         let converted: Vec<f32> = (0..num_elements)
    //             .map(|i| unsafe { std::ptr::read_unaligned(ptr.add(i)) }.to_f32())
    //             .collect();
    //         Array::from_shape_vec(IxDyn(&shape), converted)
    //             .unwrap_or_else(|_| Array::zeros(0).into_dyn())
    //     }

    //     fn _get_tensor_ptr(x: &DynValue) -> Option<(Vec<usize>, usize, *mut std::ffi::c_void)> {
    //         let shape: Vec<usize> = match x.dtype() {
    //             ort::value::ValueType::Tensor { shape, .. } => {
    //                 shape.iter().map(|&d| d as usize).collect()
    //             }
    //             _ => return None,
    //         };
    //         let num_elements: usize = shape.iter().product();
    //         if num_elements == 0 {
    //             return None;
    //         }
    //         let mut data_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    //         unsafe {
    //             let api = ort::api();
    //             let status = (api.GetTensorMutableData)(x.ptr().cast_mut(), &mut data_ptr);
    //             if !status.0.is_null() {
    //                 (api.ReleaseStatus)(status.0);
    //                 return None;
    //             }
    //         }
    //         if data_ptr.is_null() {
    //             None
    //         } else {
    //             Some((shape, num_elements, data_ptr))
    //         }
    //     }

    //     let x = match dtype {
    //         TensorElementType::Float32 => _extract_and_convert::<f32>(x, |x| x),
    //         TensorElementType::Float16 => _extract_f16_unaligned(x),
    //         TensorElementType::Bfloat16 => _extract_bf16_unaligned(x),
    //         TensorElementType::Float64 => _extract_and_convert::<f64>(x, |x| x as f32),
    //         TensorElementType::Int64 => _extract_and_convert::<i64>(x, |x| x as f32),
    //         TensorElementType::Int32 => _extract_and_convert::<i32>(x, |x| x as f32),
    //         TensorElementType::Int16 => _extract_and_convert::<i16>(x, |x| x as f32),
    //         TensorElementType::Int8 => _extract_and_convert::<i8>(x, |x| x as f32),
    //         TensorElementType::Uint64 => _extract_and_convert::<u64>(x, |x| x as f32),
    //         TensorElementType::Uint32 => _extract_and_convert::<u32>(x, |x| x as f32),
    //         TensorElementType::Uint16 => _extract_and_convert::<u16>(x, |x| x as f32),
    //         TensorElementType::Uint8 => _extract_and_convert::<u8>(x, |x| x as f32),
    //         TensorElementType::Bool => _extract_and_convert::<bool>(x, |x| x as u8 as f32),
    //         _ => return Err(anyhow::anyhow!("Unsupported ort tensor type: {:?}", dtype)),
    //     };

    //     Ok(x)
    // }

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
                        .with_fp16(self.ep.tensorrt.fp16)
                        .with_engine_cache(self.ep.tensorrt.engine_cache)
                        .with_timing_cache(self.ep.tensorrt.timing_cache)
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
                        .with_static_input_shapes(self.ep.coreml.static_input_shapes)
                .with_subgraphs(self.ep.coreml.subgraph_running)
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
                            self.ep.openvino.num_threads.unwrap_or(n_threads_available),
                        )
                        .with_dynamic_shapes(self.ep.openvino.dynamic_shapes)
                        .with_opencl_throttling(self.ep.openvino.opencl_throttling)
                        .with_qdq_optimizer(self.ep.openvino.qdq_optimizer)
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
                        .with_cann_graph(self.ep.cann.graph_inference)
                        .with_dump_graphs(self.ep.cann.dump_graphs)
                        .with_dump_om_model(self.ep.cann.dump_om_model);
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
                        .with_arena_allocator(self.ep.onednn.arena_allocator);
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
                        .with_fp16(self.ep.nnapi.fp16)
                        .with_nchw(self.ep.nnapi.nchw)
                        .with_cpu_only(self.ep.nnapi.cpu_only)
                        .with_disable_cpu(self.ep.nnapi.disable_cpu);
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
                        .with_arena_allocator(self.ep.armnn.arena_allocator);
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
                        .with_fp16(self.ep.migraphx.fp16)
                        .with_exhaustive_tune(self.ep.migraphx.exhaustive_tune);
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
            Device::Wgpu(_) => {
                #[cfg(not(feature = "wgpu"))]
                {
                    anyhow::bail!(feature_help
                        .replace("#EP", "WebGPU")
                        .replace("#FEATURE", "wgpu"));
                }
                #[cfg(feature = "wgpu")]
                {
                    let ep = ort::execution_providers::WebGPUExecutionProvider::default();
                    match ep.is_available() {
                        Ok(true) => {
                            ep.register(&mut builder).map_err(|err| {
                                anyhow::anyhow!("Failed to register WebGPU: {}", err)
                            })?;
                        }
                        _ => anyhow::bail!(compile_help.replace("#EP", "WebGPU")),
                    }
                }
            }
            _ => {
                let ep = ort::execution_providers::CPUExecutionProvider::default()
                    .with_arena_allocator(self.ep.cpu.arena_allocator);
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
        let path_ref = p.as_ref();
        let f = std::fs::read(path_ref).map_err(|err| {
            anyhow::anyhow!(
                "Failed to read ONNX file '{:?}': {}. Error: {}",
                path_ref,
                err,
                err
            )
        })?;
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
        crate::global_ts_manager().print_global_summary();
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
