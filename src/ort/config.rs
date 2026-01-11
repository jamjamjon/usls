use aksr::Builder;
use anyhow::Result;

use crate::{try_fetch_file_stem, DType, Device, EpConfig, Hub, Iiix, MinOptMax};

/// ONNX Runtime configuration with device and optimization settings.
#[derive(Builder, Debug, Clone)]
pub struct ORTConfig {
    pub file: String,
    pub external_data_file: bool,
    pub device: Device,
    pub iiixs: Vec<Iiix>,
    pub num_dry_run: usize,
    pub spec: String, // TODO
    pub dtype: DType, // For dynamically loading the model
    pub graph_opt_level: Option<u8>,
    pub num_intra_threads: Option<usize>,
    pub num_inter_threads: Option<usize>,
    pub ep: EpConfig,
}

impl Default for ORTConfig {
    fn default() -> Self {
        Self {
            file: Default::default(),
            external_data_file: false,
            device: Default::default(),
            iiixs: Default::default(),
            spec: Default::default(),
            dtype: Default::default(),
            num_dry_run: 3,
            graph_opt_level: Default::default(),
            num_intra_threads: None,
            num_inter_threads: None,
            ep: EpConfig::new(),
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

                    let parts: Vec<&str> = self.file.split('/').filter(|x| !x.is_empty()).collect();
                    if parts.len() > 1 {
                        self.file = Hub::default().try_fetch(&self.file)?;
                    } else {
                        self.file = Hub::default().try_fetch(&format!("{}/{}", name, self.file))?;
                    }

                    // try fetch external data file if it exists
                    if self.external_data_file {
                        let external_data_file = format!("{}_data", self.file);
                        tracing::info!("Trying to fetch external data file {}", external_data_file);

                        match Hub::default().try_fetch(&external_data_file) {
                            Ok(external_data_file) => {
                                tracing::info!(
                                    "Successfully fetched external data file: {}",
                                    external_data_file
                                );
                            }
                            Err(_) => {
                                tracing::warn!(
                                    "No external data file found for model {}",
                                    self.file
                                );
                            }
                        }
                    } else {
                        tracing::info!("External data file is not enabled for model {}", self.file);
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

    pub fn with_cpu_arena_allocator(mut self, x: bool) -> Self {
        self.ep.cpu.arena_allocator = x;
        self
    }

    pub fn with_openvino_dynamic_shapes(mut self, x: bool) -> Self {
        self.ep.openvino.dynamic_shapes = x;
        self
    }

    pub fn with_openvino_opencl_throttling(mut self, x: bool) -> Self {
        self.ep.openvino.opencl_throttling = x;
        self
    }

    pub fn with_openvino_qdq_optimizer(mut self, x: bool) -> Self {
        self.ep.openvino.qdq_optimizer = x;
        self
    }

    pub fn with_openvino_num_threads(mut self, x: usize) -> Self {
        self.ep.openvino.num_threads = Some(x);
        self
    }

    pub fn with_onednn_arena_allocator(mut self, x: bool) -> Self {
        self.ep.onednn.arena_allocator = x;
        self
    }

    pub fn with_tensorrt_fp16(mut self, x: bool) -> Self {
        self.ep.tensorrt.fp16 = x;
        self
    }

    pub fn with_tensorrt_engine_cache(mut self, x: bool) -> Self {
        self.ep.tensorrt.engine_cache = x;
        self
    }

    pub fn with_tensorrt_timing_cache(mut self, x: bool) -> Self {
        self.ep.tensorrt.timing_cache = x;
        self
    }

    pub fn with_coreml_static_input_shapes(mut self, x: bool) -> Self {
        self.ep.coreml.static_input_shapes = x;
        self
    }

    pub fn with_coreml_subgraph_running(mut self, x: bool) -> Self {
        self.ep.coreml.subgraph_running = x;
        self
    }

    pub fn with_cann_graph_inference(mut self, x: bool) -> Self {
        self.ep.cann.graph_inference = x;
        self
    }

    pub fn with_cann_dump_graphs(mut self, x: bool) -> Self {
        self.ep.cann.dump_graphs = x;
        self
    }

    pub fn with_cann_dump_om_model(mut self, x: bool) -> Self {
        self.ep.cann.dump_om_model = x;
        self
    }

    pub fn with_nnapi_cpu_only(mut self, x: bool) -> Self {
        self.ep.nnapi.cpu_only = x;
        self
    }

    pub fn with_nnapi_disable_cpu(mut self, x: bool) -> Self {
        self.ep.nnapi.disable_cpu = x;
        self
    }

    pub fn with_nnapi_fp16(mut self, x: bool) -> Self {
        self.ep.nnapi.fp16 = x;
        self
    }

    pub fn with_nnapi_nchw(mut self, x: bool) -> Self {
        self.ep.nnapi.nchw = x;
        self
    }

    pub fn with_armnn_arena_allocator(mut self, x: bool) -> Self {
        self.ep.armnn.arena_allocator = x;
        self
    }

    pub fn with_migraphx_fp16(mut self, x: bool) -> Self {
        self.ep.migraphx.fp16 = x;
        self
    }

    pub fn with_migraphx_exhaustive_tune(mut self, x: bool) -> Self {
        self.ep.migraphx.exhaustive_tune = x;
        self
    }

    // TODO: EP methods
}
