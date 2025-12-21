//! Hardware-specific configuration structures for different execution providers.
//!
//! This module provides specialized configuration structures for various ep
//! accelerators, promoting better separation of concerns and maintainability.

use serde::{Deserialize, Serialize};

/// CPU execution provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CpuConfig {
    /// Enable CPU arena allocator for memory management.
    pub arena_allocator: bool,
}

/// CUDA execution provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CudaConfig {
    pub memory_limit: Option<usize>,
    pub cuda_graph: bool,
    pub skip_layer_norm_strict_mode: bool,
    pub fuse_conv_bias: bool,
    pub conv_max_workspace: bool,
}

/// NVIDIA TensorRT execution provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TensorRtConfig {
    /// Enable FP16 precision for faster inference.
    pub fp16: bool,
    /// Enable engine caching to speed up subsequent runs.
    pub engine_cache: bool,
    /// Enable timing cache for optimization.
    pub timing_cache: bool,
}

/// Intel OpenVINO execution provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenVinoConfig {
    /// Enable dynamic shapes support.
    pub dynamic_shapes: bool,
    /// Enable OpenCL throttling.
    pub opencl_throttling: bool,
    /// Enable QDQ (Quantize-Dequantize) optimizer.
    pub qdq_optimizer: bool,
    /// Number of threads for OpenVINO execution.
    pub num_threads: Option<usize>,
}

/// Intel oneDNN execution provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OneDnnConfig {
    /// Enable arena allocator for memory management.
    pub arena_allocator: bool,
}

/// Apple CoreML execution provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoreMlConfig {
    /// Use static input shapes for optimization.
    pub static_input_shapes: bool,
    /// Enable subgraph running mode.
    pub subgraph_running: bool,
}

/// Huawei CANN execution provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CannConfig {
    /// Enable graph inference mode.
    pub graph_inference: bool,
    /// Enable graph dumping for debugging.
    pub dump_graphs: bool,
    /// Enable OM model dumping.
    pub dump_om_model: bool,
}

/// Android NNAPI execution provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

/// ARM NN execution provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArmNnConfig {
    /// Enable arena allocator for memory management.
    pub arena_allocator: bool,
}

/// AMD MIGraphX execution provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MiGraphXConfig {
    /// Enable FP16 precision.
    pub fp16: bool,
    /// Enable exhaustive tuning for optimization.
    pub exhaustive_tune: bool,
}

/// WebGPU execution provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebGpuConfig {
    // TODO
}

/// Unified ep configuration containing all execution provider configs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpConfig {
    /// CPU execution provider configuration.
    pub cpu: CpuConfig,
    /// CUDA execution provider configuration.
    pub cuda: CudaConfig,
    /// TensorRT execution provider configuration.
    pub tensorrt: TensorRtConfig,
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

impl EpConfig {
    /// Create a new ep configuration with sensible defaults.
    pub fn new() -> Self {
        Self {
            cpu: CpuConfig {
                arena_allocator: true,
            },
            cuda: CudaConfig {
                memory_limit: None,
                cuda_graph: false,
                skip_layer_norm_strict_mode: false,
                fuse_conv_bias: false,
                conv_max_workspace: false,
            },
            tensorrt: TensorRtConfig {
                fp16: true,
                engine_cache: true,
                timing_cache: false,
            },
            openvino: OpenVinoConfig {
                dynamic_shapes: true,
                opencl_throttling: true,
                qdq_optimizer: true,
                num_threads: None,
            },
            onednn: OneDnnConfig {
                arena_allocator: true,
            },
            coreml: CoreMlConfig {
                static_input_shapes: false,
                subgraph_running: true,
            },
            cann: CannConfig {
                graph_inference: true,
                dump_graphs: false,
                dump_om_model: true,
            },
            nnapi: NnapiConfig {
                cpu_only: false,
                disable_cpu: false,
                fp16: true,
                nchw: false,
            },
            armnn: ArmNnConfig {
                arena_allocator: true,
            },
            migraphx: MiGraphXConfig {
                fp16: true,
                exhaustive_tune: false,
            },
            webgpu: WebGpuConfig {},
        }
    }
}
