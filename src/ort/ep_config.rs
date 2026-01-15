use serde::{Deserialize, Serialize};

/// CPU execution provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuConfig {
    pub arena_allocator: bool,
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            arena_allocator: true,
        }
    }
}

/// CUDA execution provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaConfig {
    // TODO: pub memory_limit: Option<usize>,
    // TODO: cudnn_conv_algo_search
    pub cuda_graph: bool,
    pub fuse_conv_bias: bool,
    /// Configure whether the Exhaustive search can use as much memory as it needs.
    /// The default is true. When false, the memory used for the search is limited to 32 MB, which will impact its ability to find an optimal convolution algorithm.
    pub conv_max_workspace: bool,
    pub tf32: bool,
    /// Configure whether to prefer [N, H, W, C] layout operations over the default [N, C, H, W] layout.
    /// Tensor cores usually operate more efficiently with the NHWC layout, so enabling this option for convolution-heavy models on Tensor core-enabled GPUs may provide a significant performance improvement.
    pub prefer_nhwc: bool,
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            cuda_graph: false,
            fuse_conv_bias: false,
            conv_max_workspace: true,
            tf32: true,
            prefer_nhwc: true,
        }
    }
}

/// NVIDIA TensorRT execution provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRtConfig {
    pub fp16: bool,
    pub engine_cache: bool,
    pub timing_cache: bool,
    pub dump_ep_context_model: bool,
    pub builder_optimization_level: u8,
    pub max_workspace_size: usize,
}

impl Default for TensorRtConfig {
    fn default() -> Self {
        Self {
            fp16: true,
            engine_cache: true,
            timing_cache: false,
            dump_ep_context_model: false,   // TODO
            builder_optimization_level: 3,  // 3, 0-5
            max_workspace_size: 1073741824, // 1G
        }
    }
}

/// NVIDIA TensorRT-RTX execution provider configuration.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct NvRtxConfig;

/// Apple CoreML execution provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlConfig {
    /// Use static input shapes for optimization.
    pub static_input_shapes: bool,
    /// Enable subgraph running mode.
    pub subgraph_running: bool,
    /// Model format: MLProgram or NeuralNetwork.
    pub model_format: u8,
    /// Compute units: All, CPUAndGPU, CPUAndNeuralEngine, or CPUOnly.
    pub compute_units: u8,
    /// Specialization strategy: Default, FastPrediction, or FastCompilation.
    pub specialization_strategy: u8,
}

impl Default for CoreMlConfig {
    fn default() -> Self {
        Self {
            static_input_shapes: false,
            subgraph_running: true,
            model_format: 0,            // MLProgram
            compute_units: 0,           // All
            specialization_strategy: 1, // FastPrediction
        }
    }
}

/// Intel OpenVINO execution provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenVinoConfig {
    pub dynamic_shapes: bool,
    pub opencl_throttling: bool,
    pub qdq_optimizer: bool,
    pub num_threads: usize,
}

impl Default for OpenVinoConfig {
    fn default() -> Self {
        Self {
            dynamic_shapes: true,
            opencl_throttling: true,
            qdq_optimizer: true,
            num_threads: 8,
        }
    }
}

/// Intel oneDNN execution provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneDnnConfig {
    pub arena_allocator: bool,
}

impl Default for OneDnnConfig {
    fn default() -> Self {
        Self {
            arena_allocator: true,
        }
    }
}

/// Huawei CANN execution provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CannConfig {
    /// Enable graph inference mode.
    pub graph_inference: bool,
    /// Enable graph dumping for debugging.
    pub dump_graphs: bool,
    /// Enable OM model dumping.
    pub dump_om_model: bool,
}
impl Default for CannConfig {
    fn default() -> Self {
        Self {
            graph_inference: true,
            dump_graphs: false,
            dump_om_model: true,
        }
    }
}

/// Android NNAPI execution provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NnapiConfig {
    /// Force CPU-only execution.
    pub cpu_only: bool,
    /// Disable CPU fallback.
    pub disable_cpu: bool,
    /// Enable FP16 precision.
    pub fp16: bool,
    /// Use NCHW data layout.
    pub nchw: bool,
}

impl Default for NnapiConfig {
    fn default() -> Self {
        Self {
            cpu_only: false,
            disable_cpu: false,
            fp16: true,
            nchw: false,
        }
    }
}

/// ARM NN execution provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmNnConfig {
    /// Enable arena allocator for memory management.
    pub arena_allocator: bool,
}

impl Default for ArmNnConfig {
    fn default() -> Self {
        Self {
            arena_allocator: true,
        }
    }
}

/// AMD MIGraphX execution provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiGraphXConfig {
    /// Enable FP16 precision.
    pub fp16: bool,
    /// Enable exhaustive tuning for optimization.
    pub exhaustive_tune: bool,
}

impl Default for MiGraphXConfig {
    fn default() -> Self {
        Self {
            fp16: true,
            exhaustive_tune: false,
        }
    }
}

/// WebGPU execution provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebGpuConfig {}

/// Unified ep configuration containing all execution provider configs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpConfig {
    /// CPU execution provider configuration.
    pub cpu: CpuConfig,
    /// CUDA execution provider configuration.
    pub cuda: CudaConfig,
    /// TensorRT execution provider configuration.
    pub tensorrt: TensorRtConfig,
    /// NVIDIA TensorRT-RTX execution provider configuration.
    pub nvrtx: NvRtxConfig,
    /// OpenVINO execution provider configuration.
    pub openvino: OpenVinoConfig,
    /// oneDNN execution provider configuration.
    pub onednn: OneDnnConfig,
    /// CoreML execution provider configuration.
    pub coreml: CoreMlConfig,
    /// CANN execution provider configuration.
    pub cann: CannConfig,
    /// NNAPI execution provider configuration.
    pub nnapi: NnapiConfig,
    /// ARM NN execution provider configuration.
    pub armnn: ArmNnConfig,
    /// MIGraphX execution provider configuration.
    pub migraphx: MiGraphXConfig,
    /// WebGPU execution provider configuration.
    pub webgpu: WebGpuConfig,
}
