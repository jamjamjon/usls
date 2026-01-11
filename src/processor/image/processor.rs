//! Image processor for high-performance vision model preprocessing

use aksr::Builder;
use anyhow::Result;

use crate::{
    CpuTransformExecutor, Device, FromConfig, Image, ImagePlan, ImageProcessorConfig,
    ImageTensorLayout, ImageTransformInfo, TransformExecutor, XAny,
};

/// Handles pre-processing for vision and vision-language models across multiple hardware backends.
///
/// # 🖼️ ImageProcessor
///
/// A high-performance image preprocessing engine that supports resizing, normalization,
/// standardization, and layout conversion with hardware acceleration. Provides flexible
/// transformation pipelines optimized for both CPU and CUDA execution.
///
/// ## Features
///
/// - **Transform Methods**: Comprehensive image transformations with multiple algorithms
/// - **Acceleration Backends**: CPU (multi-threaded + SIMD) and CUDA (GPU memory) support
/// - **Flexible Pipeline**: Configurable transformation chains with custom ordering
/// - **Hardware Optimization**: SIMD-accelerated resizing and zero-copy GPU processing
/// - **AnyRes Support**: Dynamic resolution strategies for VLM models
///
/// ## Transform Methods
///
/// ### Resize
///
/// **Algorithms (`ResizeAlg`)**:
/// - `Bilinear`, `Bicubic`, `Lanczos3`, `SuperSampling`, `Convolution`
///
/// **Modes (`ResizeMode`)**:
/// - `FitExact`: Stretch to target dimensions (distorts aspect ratio)
/// - `FitWidth`: Fit to width, scale height proportionally
/// - `FitHeight`: Fit to height, scale width proportionally
/// - `FitAdaptive`: Maintain aspect ratio, pad remaining area
/// - `Letterbox`: Center image with padding (standard for YOLO/DETR)
///
/// ### Crop (`CropMode`)
/// - `Center`: Center crop to target dimensions
/// - `Fixed`: Extract specific sub-region `(x, y, width, height)`
///
/// ### Padding (`PadMode`)
/// - `ToMultiple`: Pad to nearest multiple of window size (e.g., SwinTransformer)
/// - `ToSize`: Pad to specific dimensions
/// - `Fixed`: Explicit padding values for top/bottom/left/right
/// - **Fill Modes**: `Constant`, `Reflect`, `Replicate`, `Wrap`
///
/// ### Other Transformations
/// - **Normalization**: Mean/Std subtraction or Min-Max scaling
/// - **Layout**: Support for `CHW` (planar) and `HWC` (interleaved) formats
/// - **AnyRes**: Dynamic resolution strategies for VLM models
///
/// ## Acceleration Backends
///
/// ### CPU
/// - Parallel multi-threaded execution via `rayon`
/// - Hardware-optimized SIMD (SSE, AVX, NEON) resizing via `fast_image_resize`
/// - Efficient memory management with minimal allocations
///
/// ### CUDA
/// - End-to-end GPU processing using `cudarc`
/// - Zero-copy operations in GPU memory
/// - Eliminates CPU-GPU transfer bottlenecks
///
/// ## Examples
///
/// ```no_run
/// use usls::{ImageProcessor, ImageProcessorConfig, ResizeModeType, ResizeFilter, FromConfig, Image};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ImageProcessorConfig::default()
///     .with_resize_mode_type(ResizeModeType::Letterbox)
///     .with_resize_filter(ResizeFilter::Lanczos3)
///     .with_normalize(true);
///
/// let mut processor = ImageProcessor::from_config(config)?;
/// let image = Image::default();
/// let result = processor.process(&[image])?;
/// # Ok(())
/// # }
/// ```
///
#[derive(Builder, Debug)]
pub struct ImageProcessor {
    pub images_transform_info: Vec<ImageTransformInfo>,
    device: Device,
    plan: ImagePlan,
    #[cfg(feature = "cuda")]
    cuda_preprocessor: Option<crate::CudaPreprocessor>,
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self {
            device: Device::Cpu(0),
            plan: ImagePlan::default(),
            images_transform_info: vec![],
            #[cfg(feature = "cuda")]
            cuda_preprocessor: None,
        }
    }
}

impl FromConfig for ImageProcessor {
    type Config = ImageProcessorConfig;

    fn from_config(config: ImageProcessorConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let cuda_preprocessor = match config.device {
            Device::Cuda(_device_id) => Some(
                crate::CudaPreprocessor::new(_device_id)
                    .map_err(|e| anyhow::anyhow!("Failed to initialize CUDA preprocessor: {e}"))?,
            ),
            _ => None,
        };

        #[cfg(not(feature = "cuda"))]
        if matches!(config.device, Device::Cuda(_)) {
            anyhow::bail!(
                "ImageProcessor device is CUDA, but crate was built without the `cuda` feature."
            );
        }

        match config.device {
            Device::Cpu(_) | Device::Cuda(_) => {}
            x => anyhow::bail!("Unsupported ImageProcessor device: {x:?}."),
        }

        // Compile config into ImagePlan (move semantics)
        let plan = config.compile_image_plan();
        let device = config.device;

        Ok(Self {
            device,
            plan,
            images_transform_info: vec![],
            #[cfg(feature = "cuda")]
            cuda_preprocessor,
        })
    }
}

impl ImageProcessor {
    /// Set image width for resize transforms.
    pub fn with_image_width(mut self, width: u32) -> Self {
        // Update any existing Resize transforms to use new width
        for transform in &mut self.plan.transforms {
            if let crate::ImageTransform::Resize(mode) = transform {
                *mode = match mode.clone() {
                    crate::ResizeMode::FitExact { height, alg, .. } => {
                        crate::ResizeMode::FitExact { width, height, alg }
                    }
                    crate::ResizeMode::FitWidth {
                        height,
                        alg,
                        padding_value,
                        ..
                    } => crate::ResizeMode::FitWidth {
                        width,
                        height,
                        alg,
                        padding_value,
                    },
                    crate::ResizeMode::FitHeight {
                        height,
                        alg,
                        padding_value,
                        ..
                    } => crate::ResizeMode::FitHeight {
                        width,
                        height,
                        alg,
                        padding_value,
                    },
                    crate::ResizeMode::FitAdaptive {
                        height,
                        alg,
                        padding_value,
                        ..
                    } => crate::ResizeMode::FitAdaptive {
                        width,
                        height,
                        alg,
                        padding_value,
                    },
                    crate::ResizeMode::Letterbox {
                        height,
                        alg,
                        padding_value,
                        ..
                    } => crate::ResizeMode::Letterbox {
                        width,
                        height,
                        alg,
                        padding_value,
                    },
                };
            }
        }
        self
    }

    /// Set image height for resize transforms.
    pub fn with_image_height(mut self, height: u32) -> Self {
        // Update any existing Resize transforms to use new height
        for transform in &mut self.plan.transforms {
            if let crate::ImageTransform::Resize(mode) = transform {
                *mode = match mode.clone() {
                    crate::ResizeMode::FitExact { width, alg, .. } => {
                        crate::ResizeMode::FitExact { width, height, alg }
                    }
                    crate::ResizeMode::FitWidth {
                        width,
                        alg,
                        padding_value,
                        ..
                    } => crate::ResizeMode::FitWidth {
                        width,
                        height,
                        alg,
                        padding_value,
                    },
                    crate::ResizeMode::FitHeight {
                        width,
                        alg,
                        padding_value,
                        ..
                    } => crate::ResizeMode::FitHeight {
                        width,
                        height,
                        alg,
                        padding_value,
                    },
                    crate::ResizeMode::FitAdaptive {
                        width,
                        alg,
                        padding_value,
                        ..
                    } => crate::ResizeMode::FitAdaptive {
                        width,
                        height,
                        alg,
                        padding_value,
                    },
                    crate::ResizeMode::Letterbox {
                        width,
                        alg,
                        padding_value,
                        ..
                    } => crate::ResizeMode::Letterbox {
                        width,
                        height,
                        alg,
                        padding_value,
                    },
                };
            }
        }
        self
    }

    /// Builder method to set AnyRes strategy for VLM models
    /// AnyRes is inserted at the beginning of transforms chain, and resize is updated
    pub fn with_dynres_strategy(mut self, strategy: Option<crate::AnyResStrategy>) -> Self {
        if let Some(s) = strategy {
            // Extract dimensions from existing resize transform if present
            let (width, height, alg) = self
                .plan
                .transforms
                .iter()
                .find_map(|t| {
                    if let crate::ImageTransform::Resize(mode) = t {
                        Some((mode.width(), mode.height(), mode.alg()))
                    } else {
                        None
                    }
                })
                .unwrap_or((640, 640, crate::ResizeAlg::default()));

            // Clear existing transforms and rebuild with AnyRes + Resize
            self.plan.transforms.clear();
            self.plan
                .insert_transform_first(crate::ImageTransform::AnyRes(s));
            // Add resize transform with extracted dimensions
            self.plan.add_transform(crate::ImageTransform::Resize(
                crate::ResizeMode::FitAdaptive {
                    width,
                    height,
                    alg,
                    padding_value: 114,
                },
            ));
        }
        self
    }

    /// Get configured image width.
    #[inline]
    pub fn image_width(&self) -> u32 {
        self.plan
            .transforms
            .iter()
            .find_map(|t| {
                if let crate::ImageTransform::Resize(mode) = t {
                    Some(mode.width())
                } else {
                    None
                }
            })
            .unwrap_or(640)
    }

    /// Get configured image height.
    #[inline]
    pub fn image_height(&self) -> u32 {
        self.plan
            .transforms
            .iter()
            .find_map(|t| {
                if let crate::ImageTransform::Resize(mode) = t {
                    Some(mode.height())
                } else {
                    None
                }
            })
            .unwrap_or(640)
    }

    /// Get image tensor layout (CHW/HWC).
    #[inline]
    pub fn image_tensor_layout(&self) -> ImageTensorLayout {
        self.plan.layout
    }

    /// Clear transform information cache.
    #[inline]
    pub fn reset_transform_info(&mut self) {
        self.images_transform_info.clear();
    }

    /// Get transform information from last processing.
    #[inline]
    pub fn transform_info(&self) -> &[ImageTransformInfo] {
        &self.images_transform_info
    }

    /// Process images through the processing pipeline and return transform info.
    ///
    /// Returns `(XAny, Vec<ImageTransformInfo>)` for VLM models that need grid info.
    pub fn process_with_info(&mut self, xs: &[Image]) -> Result<(XAny, Vec<ImageTransformInfo>)> {
        let result = self.process(xs)?;
        Ok((result, self.images_transform_info.clone()))
    }

    /// Process images through the processing pipeline.
    ///
    /// Returns `XAny` which can be either:
    /// - `Host(X)`: CPU tensor (standard ndarray)
    /// - `Device(XCuda)`: CUDA device tensor (zero-copy, CUDA feature only)
    pub fn process(&mut self, xs: &[Image]) -> Result<XAny> {
        match self.device {
            Device::Cpu(_) => {
                let executor = CpuTransformExecutor::new();
                let (x, infos) = executor.execute_plan(xs, &self.plan)?;
                self.images_transform_info = infos;
                Ok(x)
            }

            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    if let Some(ref cuda_prep) = self.cuda_preprocessor {
                        let (x, infos) = cuda_prep.execute_plan(xs, &self.plan)?;
                        self.images_transform_info = infos;
                        Ok(x)
                    } else {
                        anyhow::bail!("CUDA preprocessor not initialized")
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    anyhow::bail!("CUDA device not supported in this build")
                }
            }

            device => anyhow::bail!("Unsupported ImageProcessor device: {device:?}."),
        }
    }
}
