use aksr::Builder;
use anyhow::Result;

use crate::{try_fetch_file_stem, DType, Device, Hub, Iiix, MinOptMax};

#[derive(Builder, Debug, Clone)]
pub struct ORTConfig {
    pub file: String,
    pub device: Device,
    pub iiixs: Vec<Iiix>,
    pub num_dry_run: usize,
    pub spec: String, // TODO: move out
    pub dtype: DType, // For dynamically loading the model
    // global
    pub graph_opt_level: Option<u8>,
    pub num_intra_threads: Option<usize>,
    pub num_inter_threads: Option<usize>,
    // cpu
    pub cpu_arena_allocator: bool,
    // openvino
    pub openvino_dynamic_shapes: bool,
    pub openvino_opencl_throttling: bool,
    pub openvino_qdq_optimizer: bool,
    pub openvino_num_threads: Option<usize>,
    // onednn
    pub onednn_arena_allocator: bool,
    // tensorrt
    pub tensorrt_fp16: bool,
    pub tensorrt_engine_cache: bool,
    pub tensorrt_timing_cache: bool,
    // coreml
    pub coreml_static_input_shapes: bool,
    pub coreml_subgraph_running: bool,
    // cann
    pub cann_graph_inference: bool,
    pub cann_dump_graphs: bool,
    pub cann_dump_om_model: bool,
    // nnapi
    pub nnapi_cpu_only: bool,
    pub nnapi_disable_cpu: bool,
    pub nnapi_fp16: bool,
    pub nnapi_nchw: bool,
    // armnn
    pub armnn_arena_allocator: bool,
    // migraphx
    pub migraphx_fp16: bool,
    pub migraphx_exhaustive_tune: bool,
}

impl Default for ORTConfig {
    fn default() -> Self {
        Self {
            file: Default::default(),
            device: Default::default(),
            iiixs: Default::default(),
            spec: Default::default(),
            dtype: Default::default(),
            num_dry_run: 3,
            graph_opt_level: Default::default(),
            num_intra_threads: None,
            num_inter_threads: None,
            cpu_arena_allocator: true,
            openvino_dynamic_shapes: true,
            openvino_opencl_throttling: true,
            openvino_qdq_optimizer: true,
            openvino_num_threads: None,
            coreml_static_input_shapes: false,
            coreml_subgraph_running: true,
            tensorrt_fp16: true,
            tensorrt_engine_cache: true,
            tensorrt_timing_cache: false,
            cann_graph_inference: true,
            cann_dump_graphs: false,
            cann_dump_om_model: false,
            onednn_arena_allocator: true,
            nnapi_cpu_only: false,
            nnapi_disable_cpu: false,
            nnapi_fp16: true,
            nnapi_nchw: false,
            armnn_arena_allocator: true,
            migraphx_fp16: true,
            migraphx_exhaustive_tune: false,
        }
    }
}

impl ORTConfig {
    pub fn try_commit(mut self, name: &str) -> Result<Self> {
        // Identify the local model or fetch the remote model
        if std::path::PathBuf::from(&self.file).exists() {
            // Local
            self.spec = format!("{}/{}", name, try_fetch_file_stem(&self.file)?);
        } else {
            if self.file.is_empty() && name.is_empty() {
                anyhow::bail!(
                    "Failed to commit model. Invalid model config: neither `name` nor `file` were specified. Failed to fetch model from Hub."
                )
            }

            // Remote
            match Hub::is_valid_github_release_url(&self.file) {
                Some((owner, repo, tag, _file_name)) => {
                    let stem = try_fetch_file_stem(&self.file)?;
                    self.spec = format!("{}/{}-{}-{}-{}", name, owner, repo, tag, stem);
                    self.file = Hub::default().try_fetch(&self.file)?;
                }
                None => {
                    // append dtype to model file
                    match self.dtype {
                        d @ (DType::Auto | DType::Fp32) => {
                            if self.file.is_empty() {
                                self.file = format!("{}.onnx", d);
                            }
                        }
                        dtype => {
                            if self.file.is_empty() {
                                self.file = format!("{}.onnx", dtype);
                            } else {
                                let pos = self.file.len() - 5; // .onnx
                                let suffix = self.file.split_off(pos);
                                self.file = format!("{}-{}{}", self.file, dtype, suffix);
                            }
                        }
                    }

                    let stem = try_fetch_file_stem(&self.file)?;
                    self.spec = format!("{}/{}", name, stem);
                    self.file = Hub::default().try_fetch(&format!("{}/{}", name, self.file))?;

                    // try fetch external data file if it exists
                    match Hub::default().try_fetch(&format!("{}_data", self.file)) {
                        Ok(external_data_file) => {
                            log::debug!(
                                "Successfully fetched external data file: {}",
                                external_data_file
                            );
                        }
                        Err(_) => {
                            log::debug!("No external data file found for model {}", self.file);
                        }
                    }
                }
            }
        }

        Ok(self)
    }
}

impl ORTConfig {
    pub fn with_ixx(mut self, i: usize, ii: usize, x: MinOptMax) -> Self {
        self.iiixs.push(Iiix::from((i, ii, x)));
        self
    }

    pub fn with_batch_size(mut self, x: MinOptMax) -> Self {
        self.iiixs.push(Iiix::from((0, 0, x)));
        self
    }
}

#[macro_export]
macro_rules! impl_ort_config_methods {
    ($ty:ty, $field:ident) => {
        impl $ty {
            paste::paste! {
                pub fn [<with_ $field _file>](mut self, file: &str) -> Self {
                    self.$field = self.$field.with_file(file);
                    self
                }
                pub fn [<with_ $field _dtype>](mut self, dtype: $crate::DType) -> Self {
                    self.$field = self.$field.with_dtype(dtype);
                    self
                }
                pub fn [<with_ $field _device>](mut self, device: $crate::Device) -> Self {
                    self.$field = self.$field.with_device(device);
                    self
                }
                pub fn [<with_ $field _num_dry_run>](mut self, x: usize) -> Self {
                    self.$field = self.$field.with_num_dry_run(x);
                    self
                }
                pub fn [<with_ $field _ixx>](mut self, i: usize, ii: usize, x: $crate::MinOptMax) -> Self {
                    self.$field = self.$field.with_ixx(i, ii, x);
                    self
                }
                // global
                pub fn [<with_ $field _graph_opt_level>](mut self, x: u8) -> Self {
                    self.$field = self.$field.with_graph_opt_level(x);
                    self
                }
                pub fn [<with_ $field _num_intra_threads>](mut self, x: usize) -> Self {
                    self.$field = self.$field.with_num_intra_threads(x);
                    self
                }
                pub fn [<with_ $field _num_inter_threads>](mut self, x: usize) -> Self {
                    self.$field = self.$field.with_num_inter_threads(x);
                    self
                }
                // cpu
                pub fn [<with_ $field _cpu_arena_allocator>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_cpu_arena_allocator(x);
                    self
                }
                // openvino
                pub fn [<with_ $field _openvino_dynamic_shapes>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_openvino_dynamic_shapes(x);
                    self
                }
                pub fn [<with_ $field _openvino_opencl_throttling>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_openvino_opencl_throttling(x);
                    self
                }
                pub fn [<with_ $field _openvino_qdq_optimizer>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_openvino_qdq_optimizer(x);
                    self
                }
                pub fn [<with_ $field _openvino_num_threads>](mut self, x: usize) -> Self {
                    self.$field = self.$field.with_openvino_num_threads(x);
                    self
                }
                // onednn
                pub fn [<with_ $field _onednn_arena_allocator>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_onednn_arena_allocator(x);
                    self
                }

                // tensorrt
                pub fn [<with_ $field _tensorrt_fp16>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_tensorrt_fp16(x);
                    self
                }
                pub fn [<with_ $field _tensorrt_engine_cache>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_tensorrt_engine_cache(x);
                    self
                }
                pub fn [<with_ $field _tensorrt_timing_cache>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_tensorrt_timing_cache(x);
                    self
                }
                // coreml
                pub fn [<with_ $field _coreml_static_input_shapes>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_coreml_static_input_shapes(x);
                    self
                }
                pub fn [<with_ $field _coreml_subgraph_running>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_coreml_subgraph_running(x);
                    self
                }
                // cann
                pub fn [<with_ $field _cann_graph_inference>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_cann_graph_inference(x);
                    self
                }
                pub fn [<with_ $field _cann_dump_graphs>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_cann_dump_graphs(x);
                    self
                }
                pub fn [<with_ $field _cann_dump_om_model>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_cann_dump_om_model(x);
                    self
                }
                // nnapi
                pub fn [<with_ $field _nnapi_cpu_only>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_nnapi_cpu_only(x);
                    self
                }
                pub fn [<with_ $field _nnapi_disable_cpu>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_nnapi_disable_cpu(x);
                    self
                }
                pub fn [<with_ $field _nnapi_fp16>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_nnapi_fp16(x);
                    self
                }
                pub fn [<with_ $field _nnapi_nchw>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_nnapi_nchw(x);
                    self
                }
                // armnn
                pub fn [<with_ $field _armnn_arena_allocator>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_armnn_arena_allocator(x);
                    self
                }
                // migraphx
                pub fn [<with_ $field _migraphx_fp16>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_migraphx_fp16(x);
                    self
                }
                pub fn [<with_ $field _migraphx_exhaustive_tune>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_migraphx_exhaustive_tune(x);
                    self
                }
            }
        }
    };
}
