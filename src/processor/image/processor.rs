//! Image processing pipeline with CPU/CUDA/Metal backend support.

use aksr::Builder;
use anyhow::Result;
use ndarray::{s, Array};
use rayon::prelude::*;
use std::sync::Mutex;

#[cfg(feature = "wgpu")]
use crate::processor::image::WgpuPreprocessor;
#[cfg(feature = "cuda")]
use crate::CudaPreprocessor;
#[cfg(feature = "cuda")]
use crate::XCuda;
use crate::{
    Device, FromConfig, Image, ImagePlan, ImageProcessorConfig, ImageTensorLayout,
    ImageTransformInfo, XAny, X,
};

/// Image processing pipeline.
///
/// Handles image resizing, normalization, standardization, and layout conversion.
/// Supports CPU (via `fast_image_resize`) and CUDA backends.
#[derive(Builder, Debug)]
pub struct ImageProcessor {
    /// Device for image processing (CPU or CUDA).
    pub device: Device,
    /// Compiled processing plan (contains all static config).
    pub plan: ImagePlan,
    /// Runtime: transformation info from last process() call.
    pub images_transform_info: Vec<ImageTransformInfo>,
    /// Super Resolution: whether to pad image.
    pub pad_image: bool,
    /// Super Resolution: padding size.
    pub pad_size: usize,
    /// Super Resolution: up-scaling factor.
    pub up_scale: f32,
    /// Whether to perform resize (can skip for SR workflows).
    pub do_resize: bool,
    #[cfg(feature = "cuda")]
    cuda_preprocessor: Option<CudaPreprocessor>,
    #[cfg(feature = "wgpu")]
    wgpu_preprocessor: Option<WgpuPreprocessor>,
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self {
            device: Device::Cpu(0),
            plan: ImagePlan::default(),
            images_transform_info: vec![],
            pad_image: false,
            pad_size: 8,
            up_scale: 2.0,
            do_resize: true,
            #[cfg(feature = "cuda")]
            cuda_preprocessor: None,
            #[cfg(feature = "wgpu")]
            wgpu_preprocessor: None,
        }
    }
}

impl FromConfig for ImageProcessor {
    type Config = ImageProcessorConfig;

    fn from_config(config: ImageProcessorConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let cuda_preprocessor = match config.device {
            Device::Cuda(_device_id) => Some(
                CudaPreprocessor::new(_device_id)
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

        #[cfg(feature = "wgpu")]
        let wgpu_preprocessor = match config.device {
            Device::Wgpu(_device_id) => Some(
                WgpuPreprocessor::new(_device_id)
                    .map_err(|e| anyhow::anyhow!("Failed to initialize WGPU preprocessor: {e}"))?,
            ),
            _ => None,
        };

        #[cfg(not(feature = "wgpu"))]
        if matches!(config.device, Device::Wgpu(_)) {
            anyhow::bail!(
                "ImageProcessor device is WGPU, but crate was built without the `wgpu` feature."
            );
        }

        match config.device {
            Device::Cpu(_) | Device::Cuda(_) | Device::Wgpu(_) => {}
            x => anyhow::bail!("Unsupported ImageProcessor device: {x:?}."),
        }

        // Compile config into ImagePlan (move semantics)
        let plan = config.compile_image_plan();
        let device = config.device;
        let pad_image = config.pad_image;
        let pad_size = config.pad_size;
        let up_scale = config.up_scale;
        let do_resize = config.do_resize;

        Ok(Self {
            device,
            plan,
            images_transform_info: vec![],
            pad_image,
            pad_size,
            up_scale,
            do_resize,
            #[cfg(feature = "cuda")]
            cuda_preprocessor,
            #[cfg(feature = "wgpu")]
            wgpu_preprocessor,
        })
    }
}

impl ImageProcessor {
    /// Builder method to set image width
    pub fn with_image_width(mut self, width: u32) -> Self {
        self.plan.width = width;
        self
    }

    /// Builder method to set image height
    pub fn with_image_height(mut self, height: u32) -> Self {
        self.plan.height = height;
        self
    }

    /// Get image width from plan.
    #[inline]
    pub fn image_width(&self) -> u32 {
        self.plan.width
    }

    /// Get image height from plan.
    #[inline]
    pub fn image_height(&self) -> u32 {
        self.plan.height
    }

    /// Get image tensor layout from plan.
    #[inline]
    pub fn image_tensor_layout(&self) -> ImageTensorLayout {
        self.plan.layout
    }

    /// Clear cached image transform info.
    pub fn reset_transform_info(&mut self) {
        self.images_transform_info.clear();
    }

    /// Get last transform info.
    pub fn transform_info(&self) -> &[ImageTransformInfo] {
        &self.images_transform_info
    }

    /// Process images through the processing pipeline.
    ///
    /// Returns `XAny` which can be either:
    /// - `Host(X)`: CPU tensor (standard ndarray)
    /// - `Device(XCuda)`: CUDA device tensor (zero-copy, CUDA feature only)
    pub fn process(&mut self, xs: &[Image]) -> Result<XAny> {
        match self.device {
            Device::Cpu(_) => self.process_cpu(xs).map(XAny::from_host),
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    let cuda_prep = std::mem::take(&mut self.cuda_preprocessor);
                    let result = if let Some(ref cuda_prep) = cuda_prep {
                        self.process_cuda_device(cuda_prep, xs)
                    } else {
                        anyhow::bail!("CUDA preprocessor not initialized")
                    };
                    self.cuda_preprocessor = cuda_prep;
                    result
                }
                #[cfg(not(feature = "cuda"))]
                {
                    anyhow::bail!("CUDA device not supported in this build")
                }
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => anyhow::bail!("CUDA device not supported in this build"),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(_) => {
                #[cfg(feature = "wgpu")]
                {
                    if self.wgpu_preprocessor.is_some() {
                        let wgpu_prep = self.wgpu_preprocessor.as_ref().unwrap();
                        let images: Vec<(&[u8], u32, u32)> = xs
                            .iter()
                            .map(|img| {
                                let (w, h) = img.dimensions();
                                (img.as_raw() as &[u8], w, h)
                            })
                            .collect();

                        let (output, trans_infos) =
                            wgpu_prep.preprocess_batch(&images, &self.plan)?;
                        self.images_transform_info = trans_infos;

                        let shape: Vec<usize> = if images.len() == 1 {
                            match self.plan.layout {
                                ImageTensorLayout::NCHW => {
                                    vec![1, 3, self.plan.height as usize, self.plan.width as usize]
                                }
                                ImageTensorLayout::NHWC => {
                                    vec![1, self.plan.height as usize, self.plan.width as usize, 3]
                                }
                                ImageTensorLayout::CHW => {
                                    vec![3, self.plan.height as usize, self.plan.width as usize]
                                }
                                ImageTensorLayout::HWC => {
                                    vec![self.plan.height as usize, self.plan.width as usize, 3]
                                }
                            }
                        } else {
                            match self.plan.layout {
                                ImageTensorLayout::NCHW | ImageTensorLayout::CHW => {
                                    vec![
                                        images.len(),
                                        3,
                                        self.plan.height as usize,
                                        self.plan.width as usize,
                                    ]
                                }
                                ImageTensorLayout::NHWC | ImageTensorLayout::HWC => {
                                    vec![
                                        images.len(),
                                        self.plan.height as usize,
                                        self.plan.width as usize,
                                        3,
                                    ]
                                }
                            }
                        };

                        let arr = Array::from_shape_vec(shape.as_slice(), output)?;
                        Ok(XAny::from_host(X(arr.into_dyn())))
                    } else {
                        anyhow::bail!("WGPU preprocessor not initialized")
                    }
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    anyhow::bail!("WGPU device not supported in this build")
                }
            }
            #[cfg(not(feature = "wgpu"))]
            Device::Wgpu(_) => anyhow::bail!("WGPU device not supported in this build"),
            device => anyhow::bail!("Unsupported ImageProcessor device: {device:?}."),
        }
    }

    /// Legacy CPU-only process method (for backward compatibility).
    ///
    /// **Deprecated**: Use `process()` which returns `XAny` instead.
    #[deprecated(since = "0.2.0", note = "Use process() which returns XAny")]
    pub fn process_to_x(&mut self, xs: &[Image]) -> Result<X> {
        self.process(xs)?.as_host()
    }

    fn process_cpu(&mut self, xs: &[Image]) -> Result<X> {
        let mut x = if self.pad_image {
            if xs.len() != 1 {
                anyhow::bail!("When pad_image is true, only one image is allowed.");
            }
            let (image, images_transform_info) = xs[0].pad(self.pad_size)?;
            self.images_transform_info = vec![images_transform_info];
            image.to_ndarray()?.insert_axis(0)?
        } else if self.do_resize {
            let (x, images_transform_info) = self.par_resize(xs)?;
            self.images_transform_info = images_transform_info;
            x
        } else {
            anyhow::bail!(
                "When pad_image and do_resize are both false, at least one image is required."
            );
        };

        if self.plan.normalize {
            x = x.normalize(0., 255.)?;
        }

        if let (Some(mean), Some(std)) = (self.plan.mean.as_ref(), self.plan.std.as_ref()) {
            x = x.standardize(mean, std, 3)?;
        }

        if self.plan.layout.is_chw() {
            x = x.nhwc2nchw()?;
        }

        if self.plan.unsigned {
            x = x.unsigned();
        }

        Ok(x)
    }

    #[cfg(feature = "cuda")]
    fn process_cuda_device(&mut self, cuda_prep: &CudaPreprocessor, xs: &[Image]) -> Result<XAny> {
        if xs.is_empty() {
            anyhow::bail!("No input images provided.");
        }

        let images: Vec<(&[u8], u32, u32)> = xs
            .iter()
            .map(|img| {
                let (w, h) = img.dimensions();
                (img.as_raw() as &[u8], w, h)
            })
            .collect();

        let (device_buffer, shape, trans_infos) =
            cuda_prep.preprocess_batch_device(&images, &self.plan)?;
        self.images_transform_info = trans_infos;

        let cuda_tensor = XCuda::new(
            device_buffer,
            shape,
            cuda_prep.stream(),
            cuda_prep.device_id(),
        );

        Ok(XAny::from_device(cuda_tensor))
    }

    /// Parallel resize images.
    pub fn par_resize(&self, xs: &[Image]) -> Result<(X, Vec<ImageTransformInfo>)> {
        // TODO: deprecated build_resizer_filter()
        // let filter_name = self.filter_name();
        let filter_name = self
            .plan
            .resize_alg
            .filter()
            .ok_or(anyhow::anyhow!("Failed to get filter name."))?
            .to_string();
        let width = self.plan.width;
        let height = self.plan.height;
        let resize_mode = &self.plan.resize_mode;
        let padding_value = self.plan.padding_value;

        match xs.len() {
            0 => anyhow::bail!("Found no input images."),
            1 => {
                let (image, trans_info) = xs[0].resize_with_info(
                    width,
                    height,
                    &filter_name,
                    resize_mode,
                    padding_value,
                )?;

                let y = image.to_ndarray()?.insert_axis(0)?;
                Ok((y, vec![trans_info]))
            }
            _ => {
                let ys = Mutex::new(
                    Array::zeros((xs.len(), height as usize, width as usize, 3)).into_dyn(),
                );

                let results: Result<Vec<ImageTransformInfo>> = xs
                    .par_iter()
                    .enumerate()
                    .map(|(idx, x)| {
                        let (image, trans_info) = x.resize_with_info(
                            width,
                            height,
                            &filter_name,
                            resize_mode,
                            padding_value,
                        )?;

                        let y = image.to_ndarray()?;
                        {
                            let mut ys_guard = ys
                                .lock()
                                .map_err(|e| anyhow::anyhow!("Mutex lock error: {e}"))?;
                            ys_guard.slice_mut(s![idx, .., .., ..]).assign(&y);
                        }

                        Ok(trans_info)
                    })
                    .collect();

                let ys_inner = ys
                    .into_inner()
                    .map_err(|e| anyhow::anyhow!("Mutex into_inner error: {e}"))?;

                Ok((ys_inner.into(), results?))
            }
        }
    }
}
