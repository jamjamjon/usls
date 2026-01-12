//! Configuration system for model inference and processing pipelines
//!
//! A flexible, type-safe configuration system for managing model engines, processors, and inference parameters.

use std::collections::HashMap;

use aksr::Builder;

use crate::{ImageProcessorConfig, InferenceParams, Module, ORTConfig, Scale, Task, Version};

/// Flexible, type-safe configuration system for model engines, processors, and inference parameters.
///
/// # 🔩 Configuration System
///
/// A comprehensive configuration management system that handles model inference setups,
/// processing pipelines, and runtime parameters. Provides a unified interface for
/// configuring vision and vision-language models with their respective processors.
///
/// ## Features
///
/// - **Dynamic Module Management**: Store modules in `HashMap<Module, ORTConfig>` for flexible configuration
/// - **Consolidated Parameters**: All inference parameters unified in `InferenceParams` struct
/// - **Move Semantics**: Zero-copy module consumption with `take_module()` for efficient resource management
/// - **Auto-resolution**: Automatic model file naming and path resolution
/// - **Validation**: Configuration validation during commit
/// - **Model Downloads**: Downloads models from remote repositories if needed
/// - **YOLO Models (Special Case)**: Auto-generates filenames from version/scale/task
///   (e.g., `v8-n-det.onnx`) - **Note**: Only YOLO uses this naming pattern
///
/// ## Architecture
///
/// ```text
/// Config
/// ├── Basics
/// │   ├── name: &'static str           // Model identifier
/// │   ├── version: Option<Version>     // Model version
/// │   ├── task: Option<Task>           // Inference task type
/// │   └── scale: Option<Scale>         // Model scale/size
/// ├── Modules
/// │   └── HashMap<Module, ORTConfig>   // Dynamic module registry
/// ├── Processors
/// │   ├── image_processor: ImageProcessorConfig
/// │   └── text_processor: TextProcessorConfig (VLM feature)
/// └── Inference
///     └── inference: InferenceParams   // Runtime parameters
/// ```
///
/// ## Pre-configured Models
///
/// The system provides pre-defined configurations for popular models:
///
/// ### YOLO Models
/// - `Config::yolo().with_task(Task::ObjectDetection).with_version(8).with_scale(Scale::N)` - YOLOv8n Detect
/// - `Config::yolo().with_task(Task::InstanceSegmentation).with_version(11).with_scale(Scale::M)` - YOLO11m Segement
///
/// ### Vision Models
/// - `Config::rfdetr_nano()` - Real-time detection
/// - `Config::depth_anything_small()` - Depth estimation
///
/// ### Vision-Language Models
/// - `Config::smolvlm()` - Small VLM
/// - `Config::moondream2()` - Efficient VLM
///
/// # Examples
///
/// ## Basic Usage
/// ```no_run
/// use usls::{Config, Device, DType};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = Config::rfdetr_nano()
///     .with_model_dtype(DType::Fp16)
///     .with_model_device(Device::Cuda(0))
///     .with_image_processor_device(Device::Cuda(0))
///     .with_batch_size_all_min_opt_max(1, 1, 8)
///     .commit()?;  // Resolve paths and finalize
/// # Ok(())
/// # }
/// ```
///
/// ## YOLO Special Case
///
/// ```no_run
/// use usls::{Config, Device, Task, DType, Scale};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = Config::yolo()
///     .with_task(Task::ObjectDetection)
///     .with_version(8.into())
///     .with_scale(Scale::N)
///     .with_model_dtype(DType::Fp16)
///     .with_model_device(Device::Cuda(0))
///     .with_image_processor_device(Device::Cuda(0))
///     .commit()?;
/// # Ok(())
/// # }
/// ```
///
#[derive(Builder, Debug, Clone, Default)]
pub struct Config {
    // Basics
    pub(crate) name: &'static str,
    pub(crate) version: Option<Version>,
    pub(crate) task: Option<Task>,
    pub(crate) scale: Option<Scale>,

    // Modules (dynamic registry)
    pub(crate) modules: HashMap<Module, ORTConfig>,

    // Processors
    pub(crate) image_processor: ImageProcessorConfig,
    #[cfg(feature = "vlm")]
    pub(crate) text_processor: crate::TextProcessorConfig,

    // Inference Parameters
    pub(crate) inference: InferenceParams,
}

impl Config {
    /// Finalize and validate all module configurations.
    ///
    /// Resolves model file paths, downloads missing models, and validates the setup.
    /// Must be called before using the configuration for inference.
    pub fn commit(mut self) -> anyhow::Result<Self> {
        // Special case for YOLO: generate file name from version/scale/task
        if self.name == "yolo" {
            if let Some(model_config) = self.modules.get_mut(&Module::Model) {
                if model_config.file.is_empty() {
                    let mut y = String::new();
                    if let Some(x) = self.version {
                        y.push_str(&x.to_string());
                    }
                    if let Some(ref x) = self.scale {
                        y.push_str(&format!("-{}", x));
                    }
                    if let Some(ref x) = self.task {
                        y.push_str(&format!("-{}", x.yolo_str()));
                    }
                    y.push_str(".onnx");
                    model_config.file = y;
                }
            }
        }

        // Commit each module config
        let name = self.name;
        for (module, config) in self.modules.iter_mut() {
            if !config.file.is_empty() {
                // 1. Feature availability pre-check
                match config.device {
                    crate::Device::Cuda(_) => {
                        if !cfg!(feature = "cuda") {
                            anyhow::bail!("CUDA execution provider for module ({:?}) requires the 'cuda' feature. \n\
                            Consider enabling it by adding '-F cuda'.\n\
                            If you also need CUDA image processing, use '-F cuda-full' or a version-specific feature like '-F cuda-12040'.", module);
                        }
                    }
                    crate::Device::TensorRt(_) => {
                        if !cfg!(feature = "tensorrt") {
                            anyhow::bail!("TensorRT execution provider for module ({:?}) requires the 'tensorrt' feature. \n\
                            Consider enabling it by adding '-F tensorrt'. \n\
                            If you also need CUDA image processing, use '-F tensorrt-full' or a version-specific feature like '-F tensorrt-cuda-12040'.", module);
                        }
                    }
                    _ => {}
                }

                *config = std::mem::take(config).try_commit(name)?;

                // 2. Device compatibility check
                let p_dev = self.image_processor.device;
                let m_dev = config.device;

                // ImageProcessor only supports CPU or CUDA
                match p_dev {
                    crate::Device::Cpu(_) => {
                        // ImageProcessor CPU - no special checks needed
                    }
                    crate::Device::Cuda(cuda_id) => {
                        // ImageProcessor CUDA requires 'cuda-runtime' feature
                        if !cfg!(feature = "cuda-runtime") {
                            anyhow::bail!("GPU image processing requires CUDA runtime libraries. \n\
                            Consider enabling it by adding '-F cuda-full', '-F tensorrt-full', or a version-specific feature like '-F cuda-12040' or '-F tensorrt-cuda-12040'.");
                        }

                        // ImageProcessor CUDA requires ORT GPU
                        match m_dev {
                            crate::Device::Cpu(_) => {
                                anyhow::bail!(
                                    "Device mismatch: ImageProcessor is on CUDA({}) but Module ({:?}) is on CPU. \n\
                                    CUDA image processing requires CUDA or TensorRT execution provider for model inference. \n\
                                    Consider enabling it by adding '-F cuda-full', '-F tensorrt-full', or a version-specific feature like '-F cuda-12040' or '-F tensorrt-cuda-12040'.",
                                    cuda_id, module
                                );
                            }
                            crate::Device::Cuda(model_cuda_id) => {
                                // Check GPU ID consistency
                                if cuda_id != model_cuda_id {
                                    anyhow::bail!(
                                        "Device ID mismatch: ImageProcessor is on CUDA({}) but Module ({:?}) is on CUDA({}). \n\
                                        They must use the same GPU ID to avoid performance loss and illegal memory access.",
                                        cuda_id, module, model_cuda_id
                                    );
                                }
                            }
                            crate::Device::TensorRt(tensorrt_id) => {
                                // Check GPU ID consistency
                                if cuda_id != tensorrt_id {
                                    anyhow::bail!(
                                        "Device ID mismatch: ImageProcessor is on CUDA({}) but Module ({:?}) is on TensorRT({}). \n\
                                        They must use the same GPU ID to avoid performance loss and illegal memory access.",
                                        cuda_id, module, tensorrt_id
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {
                        anyhow::bail!("Unsupported ImageProcessor device: {:?}. Only CUDA and CPU are supported.", p_dev);
                    }
                }
            }
        }

        Ok(self)
    }

    /// Utility function to load vocabulary from text file
    pub fn load_txt_into_vec(f: &str) -> Vec<String> {
        crate::Hub::default()
            .try_fetch(f)
            .ok()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .map(|content| content.lines().map(|line| line.to_string()).collect())
            .unwrap_or_else(Vec::new)
    }
}
