//! CUDA context management for processing.
//!
//! This module provides GPU-accelerated image processing using cudarc.

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::{Arc, Mutex};

use super::*;
use crate::{
    compute_convolution_1d, ImagePlan, ImageTensorLayout, ImageTransformInfo, ResizeAlg, ResizeMode,
};

/// CUDA source code for processing kernels (embedded at compile time).
const PREPROCESS_CUDA_SRC: &str = concat!(
    include_str!("kernels/utils.cu"),
    include_str!("kernels/conv_resize.cu"),
    include_str!("kernels/postprocess.cu"),
    include_str!("kernels/transforms.cu"),
);

/// CUDA preprocessor context managing device, streams, and kernels.
/// This implements TransformExecutor for CUDA backend.
pub struct CudaPreprocessor {
    ctx: Arc<CudaContext>,
    kernel_conv_vert_u8x3: CudaFunction,
    kernel_conv_horiz_u8x3: CudaFunction,
    kernel_resize_nearest_u8x3: CudaFunction,
    kernel_post_u8_to_nchw_f32: CudaFunction,
    kernel_post_u8_to_nhwc_f32: CudaFunction,
    kernel_pad_constant: CudaFunction,
    kernel_pad_reflect: CudaFunction,
    kernel_pad_replicate: CudaFunction,
    kernel_pad_wrap: CudaFunction,
    kernel_pad_fixed_constant: CudaFunction,
    kernel_pad_fixed_reflect: CudaFunction,
    kernel_pad_fixed_replicate: CudaFunction,
    kernel_pad_fixed_wrap: CudaFunction,
    kernel_crop_center: CudaFunction,
    conv_cache: Mutex<ConvCoeffCache>,
    buffer_pool: Mutex<DeviceBufferPool>,
}

impl std::fmt::Debug for CudaPreprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaPreprocessor")
            .field("is_ready", &self.is_ready())
            .finish()
    }
}

impl crate::TransformExecutor for CudaPreprocessor {
    /// Execute image processing plan.
    ///
    /// # Implementation Strategy
    /// - **Resize-only plans**: Use optimized CUDA kernels, return device tensor (zero-copy)
    /// - **Pad/Crop transforms**: Use CUDA kernels for GPU-accelerated processing
    /// - **AnyRes transforms**: Delegate to CPU (complex patch extraction logic)
    fn execute_plan(
        &self,
        images: &[crate::Image],
        plan: &ImagePlan,
    ) -> Result<(crate::XAny, Vec<crate::ImageTransformInfo>)> {
        plan.validate()?;

        if images.is_empty() {
            anyhow::bail!("No input images provided");
        }

        // Check for AnyRes - delegate to CPU as it's complex
        let has_dynres = plan
            .transforms
            .iter()
            .any(|t| matches!(t, crate::ImageTransform::AnyRes(_)));

        if has_dynres {
            let cpu_executor = crate::CpuTransformExecutor::new();
            return cpu_executor.execute_plan(images, plan);
        }

        let is_resize_only = plan.transforms.len() == 1
            && matches!(
                plan.transforms.first(),
                Some(crate::ImageTransform::Resize(_))
            );

        if is_resize_only {
            let images_data: Vec<(&[u8], u32, u32)> = images
                .iter()
                .map(|img| {
                    let (w, h) = img.dimensions();
                    (img.as_raw() as &[u8], w, h)
                })
                .collect();

            let (device_buffer, shape, trans_infos) =
                self.preprocess_batch_device(&images_data, plan)?;

            let cuda_tensor =
                crate::XCuda::new(device_buffer, shape, self.stream(), self.device_id());

            return Ok((crate::XAny::from_device(cuda_tensor), trans_infos));
        }

        // Single image path (Pad only supports single image)
        if images.len() != 1 {
            let cpu_executor = crate::CpuTransformExecutor::new();
            return cpu_executor.execute_plan(images, plan);
        }

        // Single image: execute transforms sequentially on CUDA
        self.execute_transforms_cuda(&images[0], plan)
    }
}

impl CudaPreprocessor {
    /// Create a new CUDA preprocessor context.
    pub fn new(device_id: usize) -> Result<Self> {
        let ctx = CudaContext::new(device_id).map_err(driver_err)?;

        // Compile CUDA source to PTX at runtime
        let ptx = compile_ptx(PREPROCESS_CUDA_SRC)
            .map_err(|e| anyhow::anyhow!("Failed to compile CUDA kernels: {e:?}"))?;

        // Load PTX module and get kernel functions
        let module = ctx.load_module(ptx).map_err(driver_err)?;

        let kernel_conv_vert_u8x3 = module.load_function("conv_vert_u8x3").map_err(driver_err)?;
        let kernel_conv_horiz_u8x3 = module
            .load_function("conv_horiz_u8x3")
            .map_err(driver_err)?;
        let kernel_resize_nearest_u8x3 = module
            .load_function("resize_nearest_u8x3")
            .map_err(driver_err)?;
        let kernel_post_u8_to_nchw_f32 = module
            .load_function("postprocess_pad_u8_to_nchw_f32")
            .map_err(driver_err)?;
        let kernel_post_u8_to_nhwc_f32 = module
            .load_function("postprocess_pad_u8_to_nhwc_f32")
            .map_err(driver_err)?;
        let kernel_pad_constant = module
            .load_function("pad_to_multiple_constant")
            .map_err(driver_err)?;
        let kernel_pad_reflect = module
            .load_function("pad_to_multiple_reflect")
            .map_err(driver_err)?;
        let kernel_pad_replicate = module
            .load_function("pad_to_multiple_replicate")
            .map_err(driver_err)?;
        let kernel_pad_wrap = module
            .load_function("pad_to_multiple_wrap")
            .map_err(driver_err)?;
        let kernel_pad_fixed_constant = module
            .load_function("pad_fixed_constant")
            .map_err(driver_err)?;
        let kernel_pad_fixed_reflect = module
            .load_function("pad_fixed_reflect")
            .map_err(driver_err)?;
        let kernel_pad_fixed_replicate = module
            .load_function("pad_fixed_replicate")
            .map_err(driver_err)?;
        let kernel_pad_fixed_wrap = module.load_function("pad_fixed_wrap").map_err(driver_err)?;
        let kernel_crop_center = module.load_function("crop_center").map_err(driver_err)?;

        Ok(Self {
            ctx,
            kernel_conv_vert_u8x3,
            kernel_conv_horiz_u8x3,
            kernel_resize_nearest_u8x3,
            kernel_post_u8_to_nchw_f32,
            kernel_post_u8_to_nhwc_f32,
            kernel_pad_constant,
            kernel_pad_reflect,
            kernel_pad_replicate,
            kernel_pad_wrap,
            kernel_pad_fixed_constant,
            kernel_pad_fixed_reflect,
            kernel_pad_fixed_replicate,
            kernel_pad_fixed_wrap,
            kernel_crop_center,
            conv_cache: Mutex::new(ConvCoeffCache::new(32)),
            buffer_pool: Mutex::new(DeviceBufferPool::new()),
        })
    }

    pub fn is_ready(&self) -> bool {
        true
    }

    fn get_or_create_vert_coeffs(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
        key: ConvKey,
    ) -> Result<Arc<DeviceConvCoeffs>> {
        {
            let cache = self
                .conv_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("Conv coeff cache lock poisoned"))?;
            if let Some(entry) = cache.vert.get(&key) {
                return Ok(Arc::clone(entry));
            }
        }

        let vert = compute_convolution_1d(
            key.in_size,
            key.out_size,
            key.filter,
            key.adaptive_kernel_size,
        );

        let mut d_starts = unsafe {
            stream
                .alloc::<i32>(vert.starts().len())
                .map_err(driver_err)?
        };
        let mut d_sizes = unsafe {
            stream
                .alloc::<i32>(vert.sizes().len())
                .map_err(driver_err)?
        };
        let mut d_offsets = unsafe {
            stream
                .alloc::<i32>(vert.offsets().len())
                .map_err(driver_err)?
        };
        let mut d_coeffs = unsafe {
            stream
                .alloc::<i16>(vert.coeffs().len())
                .map_err(driver_err)?
        };
        stream
            .memcpy_htod(vert.starts(), &mut d_starts)
            .map_err(driver_err)?;
        stream
            .memcpy_htod(vert.sizes(), &mut d_sizes)
            .map_err(driver_err)?;
        stream
            .memcpy_htod(vert.offsets(), &mut d_offsets)
            .map_err(driver_err)?;
        stream
            .memcpy_htod(vert.coeffs(), &mut d_coeffs)
            .map_err(driver_err)?;

        let entry = Arc::new(DeviceConvCoeffs {
            d_starts: Arc::new(d_starts),
            d_sizes: Arc::new(d_sizes),
            d_offsets: Arc::new(d_offsets),
            d_coeffs: Arc::new(d_coeffs),
            precision: vert.precision() as i32,
        });

        let mut cache = self
            .conv_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("Conv coeff cache lock poisoned"))?;
        cache.evict_if_needed();
        cache.vert.insert(key, Arc::clone(&entry));
        Ok(entry)
    }

    fn get_or_create_horiz_coeffs(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
        key: ConvKey,
    ) -> Result<Arc<DeviceHorizConvCoeffs>> {
        {
            let cache = self
                .conv_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("Conv coeff cache lock poisoned"))?;
            if let Some(entry) = cache.horiz.get(&key) {
                return Ok(Arc::clone(entry));
            }
        }

        let horiz = compute_convolution_1d(
            key.in_size,
            key.out_size,
            key.filter,
            key.adaptive_kernel_size,
        );
        let x_first = *horiz
            .starts()
            .first()
            .ok_or_else(|| anyhow::anyhow!("Empty horizontal convolution starts"))?;
        let last_idx = horiz.starts().len() - 1;
        let x_last = horiz.starts()[last_idx] + horiz.sizes()[last_idx];
        let temp_width_i32 = x_last - x_first;
        if temp_width_i32 <= 0 {
            anyhow::bail!("Invalid temp width computed: {temp_width_i32}");
        }
        let temp_width = temp_width_i32 as u32;
        let h_starts_shifted: Vec<i32> = horiz.starts().iter().map(|s| *s - x_first).collect();

        let mut d_starts = unsafe {
            stream
                .alloc::<i32>(h_starts_shifted.len())
                .map_err(driver_err)?
        };
        let mut d_sizes = unsafe {
            stream
                .alloc::<i32>(horiz.sizes().len())
                .map_err(driver_err)?
        };
        let mut d_offsets = unsafe {
            stream
                .alloc::<i32>(horiz.offsets().len())
                .map_err(driver_err)?
        };
        let mut d_coeffs = unsafe {
            stream
                .alloc::<i16>(horiz.coeffs().len())
                .map_err(driver_err)?
        };
        stream
            .memcpy_htod(&h_starts_shifted, &mut d_starts)
            .map_err(driver_err)?;
        stream
            .memcpy_htod(horiz.sizes(), &mut d_sizes)
            .map_err(driver_err)?;
        stream
            .memcpy_htod(horiz.offsets(), &mut d_offsets)
            .map_err(driver_err)?;
        stream
            .memcpy_htod(horiz.coeffs(), &mut d_coeffs)
            .map_err(driver_err)?;

        let entry = Arc::new(DeviceHorizConvCoeffs {
            coeffs: DeviceConvCoeffs {
                d_starts: Arc::new(d_starts),
                d_sizes: Arc::new(d_sizes),
                d_offsets: Arc::new(d_offsets),
                d_coeffs: Arc::new(d_coeffs),
                precision: horiz.precision() as i32,
            },
            x_first,
            temp_width,
        });

        let mut cache = self
            .conv_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("Conv coeff cache lock poisoned"))?;
        cache.evict_if_needed();
        cache.horiz.insert(key, Arc::clone(&entry));
        Ok(entry)
    }

    /// Preprocess a single image on GPU.
    #[allow(dead_code)]
    pub fn preprocess_image(
        &self,
        input: &[u8],
        plan: &ImagePlan,
        in_width: u32,
        in_height: u32,
    ) -> Result<(Vec<f32>, ImageTransformInfo)> {
        let images = [(input, in_width, in_height)];
        let (out, mut infos) = self.preprocess_batch_with_plan(&images, plan)?;
        let info = infos
            .pop()
            .ok_or_else(|| anyhow::anyhow!("CUDA preprocess returned no ImageTransformInfo"))?;
        Ok((out, info))
    }

    /// Preprocess a single image using CudaImageProcessContext.
    #[allow(clippy::too_many_arguments)]
    fn preprocess_image_with_context(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
        input: &[u8],
        ctx: &CudaImageProcessContext,
        d_output: &mut cudarc::driver::CudaViewMut<f32>,
        d_mean: Option<&cudarc::driver::CudaSlice<f32>>,
        d_std: Option<&cudarc::driver::CudaSlice<f32>>,
        buffer_pool: &mut DeviceBufferPool,
    ) -> Result<ImageTransformInfo> {
        let (resized_w, resized_h, pad_left, pad_top, scale) =
            Self::calculate_resize_params_ctx(ctx);

        let resized_w_u32 = resized_w.max(0) as u32;
        let resized_h_u32 = resized_h.max(0) as u32;

        let (scale_w, scale_h) = match ctx.resize_mode() {
            ResizeMode::FitExact { .. } => (
                ctx.out_width() as f32 / ctx.in_width as f32,
                ctx.out_height() as f32 / ctx.in_height as f32,
            ),
            ResizeMode::FitWidth { .. } => (1.0, scale),
            ResizeMode::FitHeight { .. } => (scale, 1.0),
            ResizeMode::FitAdaptive { .. } | ResizeMode::Letterbox { .. } => (scale, scale),
        };

        let trans_info = ImageTransformInfo::default()
            .with_width_src(ctx.in_width)
            .with_height_src(ctx.in_height)
            .with_width_dst(ctx.out_width())
            .with_height_dst(ctx.out_height())
            .with_width_scale(scale_w)
            .with_height_scale(scale_h)
            .with_width_pad(pad_left as f32)
            .with_height_pad(pad_top as f32);

        let mut d_input = buffer_pool.take_or_alloc_input(stream, input.len())?;
        let pinned_input = buffer_pool.fill_pinned_input(input)?;
        stream
            .memcpy_htod(pinned_input, &mut d_input)
            .map_err(driver_err)?;

        if resized_w_u32 == 0 || resized_h_u32 == 0 {
            anyhow::bail!("Invalid resized size: {resized_w}x{resized_h}");
        }

        let block_x = 16u32;
        let block_y = 16u32;
        let in_h_i32 = ctx.in_height as i32;
        let in_w_i32 = ctx.in_width as i32;
        let resized_h_i32 = resized_h_u32 as i32;
        let resized_w_i32 = resized_w_u32 as i32;

        let resized_size = (resized_w_u32 * resized_h_u32 * 3) as usize;
        let mut d_resized = buffer_pool.take_or_alloc_resized(stream, resized_size)?;

        match ctx.resize_alg() {
            ResizeAlg::Nearest => {
                let cfg = LaunchConfig {
                    grid_dim: (
                        resized_w_u32.div_ceil(block_x),
                        resized_h_u32.div_ceil(block_y),
                        1,
                    ),
                    block_dim: (block_x, block_y, 1),
                    shared_mem_bytes: 0,
                };

                unsafe {
                    stream
                        .launch_builder(&self.kernel_resize_nearest_u8x3)
                        .arg(&d_input)
                        .arg(&mut d_resized)
                        .arg(&in_h_i32)
                        .arg(&in_w_i32)
                        .arg(&resized_h_i32)
                        .arg(&resized_w_i32)
                        .launch(cfg)
                }
                .map_err(driver_err)?;
            }
            ResizeAlg::Convolution(filter) | ResizeAlg::Interpolation(filter) => {
                let adaptive_kernel_size = matches!(ctx.resize_alg(), ResizeAlg::Convolution(_));

                let vert = self.get_or_create_vert_coeffs(
                    stream,
                    ConvKey {
                        in_size: ctx.in_height,
                        out_size: resized_h_u32,
                        filter,
                        adaptive_kernel_size,
                    },
                )?;
                let horiz = self.get_or_create_horiz_coeffs(
                    stream,
                    ConvKey {
                        in_size: ctx.in_width,
                        out_size: resized_w_u32,
                        filter,
                        adaptive_kernel_size,
                    },
                )?;

                let temp_width = horiz.temp_width;
                let offset_x_i32 = horiz.x_first;
                let v_precision: i32 = vert.precision;
                let h_precision: i32 = horiz.coeffs.precision;

                let temp_size = (temp_width * resized_h_u32 * 3) as usize;
                let mut d_temp = buffer_pool.take_or_alloc_temp(stream, temp_size)?;

                let cfg_v = LaunchConfig {
                    grid_dim: (
                        temp_width.div_ceil(block_x),
                        resized_h_u32.div_ceil(block_y),
                        1,
                    ),
                    block_dim: (block_x, block_y, 1),
                    shared_mem_bytes: 0,
                };
                let out_h_i32 = resized_h_u32 as i32;
                let out_w_i32 = temp_width as i32;

                unsafe {
                    stream
                        .launch_builder(&self.kernel_conv_vert_u8x3)
                        .arg(&d_input)
                        .arg(&mut d_temp)
                        .arg(&in_h_i32)
                        .arg(&in_w_i32)
                        .arg(&out_h_i32)
                        .arg(&out_w_i32)
                        .arg(&offset_x_i32)
                        .arg(&*vert.d_starts)
                        .arg(&*vert.d_sizes)
                        .arg(&*vert.d_offsets)
                        .arg(&*vert.d_coeffs)
                        .arg(&v_precision)
                        .launch(cfg_v)
                }
                .map_err(driver_err)?;

                let cfg_h = LaunchConfig {
                    grid_dim: (
                        resized_w_u32.div_ceil(block_x),
                        resized_h_u32.div_ceil(block_y),
                        1,
                    ),
                    block_dim: (block_x, block_y, 1),
                    shared_mem_bytes: 0,
                };
                let height_i32 = resized_h_u32 as i32;
                let in_w_h_i32 = temp_width as i32;
                let out_w_h_i32 = resized_w_u32 as i32;

                unsafe {
                    stream
                        .launch_builder(&self.kernel_conv_horiz_u8x3)
                        .arg(&d_temp)
                        .arg(&mut d_resized)
                        .arg(&height_i32)
                        .arg(&in_w_h_i32)
                        .arg(&out_w_h_i32)
                        .arg(&*horiz.coeffs.d_starts)
                        .arg(&*horiz.coeffs.d_sizes)
                        .arg(&*horiz.coeffs.d_offsets)
                        .arg(&*horiz.coeffs.d_coeffs)
                        .arg(&h_precision)
                        .launch(cfg_h)
                }
                .map_err(driver_err)?;

                buffer_pool.temp = Some(d_temp);
            }
            ResizeAlg::SuperSampling(_, _) => {
                anyhow::bail!("CUDA backend does not support SuperSampling")
            }
        }

        // Postprocess
        let normalize_scale = if ctx.normalize() {
            1.0f32 / 255.0f32
        } else {
            1.0f32
        };
        let apply_unsigned: i32 = if ctx.unsigned() { 1 } else { 0 };
        let pad_top_i32 = pad_top;
        let pad_left_i32 = pad_left;
        let padding_value_u8 = ctx.padding_value();
        let out_h_full_i32 = ctx.out_height() as i32;
        let out_w_full_i32 = ctx.out_width() as i32;
        let resized_h_i32 = resized_h_u32 as i32;
        let resized_w_i32 = resized_w_u32 as i32;
        let null_ptr: u64 = 0;
        let cfg_post = LaunchConfig {
            grid_dim: (
                ctx.out_width().div_ceil(block_x),
                ctx.out_height().div_ceil(block_y),
                1,
            ),
            block_dim: (block_x, block_y, 1),
            shared_mem_bytes: 0,
        };

        match ctx.layout() {
            ImageTensorLayout::NCHW | ImageTensorLayout::CHW => {
                let mut builder = stream.launch_builder(&self.kernel_post_u8_to_nchw_f32);
                builder
                    .arg(&d_resized)
                    .arg(d_output)
                    .arg(&resized_h_i32)
                    .arg(&resized_w_i32)
                    .arg(&out_h_full_i32)
                    .arg(&out_w_full_i32)
                    .arg(&pad_top_i32)
                    .arg(&pad_left_i32)
                    .arg(&padding_value_u8)
                    .arg(&normalize_scale);
                if let Some(mean) = d_mean {
                    builder.arg(mean);
                } else {
                    builder.arg(&null_ptr);
                }
                if let Some(std) = d_std {
                    builder.arg(std);
                } else {
                    builder.arg(&null_ptr);
                }
                builder.arg(&apply_unsigned);
                unsafe { builder.launch(cfg_post) }.map_err(driver_err)?;
            }
            ImageTensorLayout::NHWC | ImageTensorLayout::HWC => {
                let mut builder = stream.launch_builder(&self.kernel_post_u8_to_nhwc_f32);
                builder
                    .arg(&d_resized)
                    .arg(d_output)
                    .arg(&resized_h_i32)
                    .arg(&resized_w_i32)
                    .arg(&out_h_full_i32)
                    .arg(&out_w_full_i32)
                    .arg(&pad_top_i32)
                    .arg(&pad_left_i32)
                    .arg(&padding_value_u8)
                    .arg(&normalize_scale);
                if let Some(mean) = d_mean {
                    builder.arg(mean);
                } else {
                    builder.arg(&null_ptr);
                }
                if let Some(std) = d_std {
                    builder.arg(std);
                } else {
                    builder.arg(&null_ptr);
                }
                builder.arg(&apply_unsigned);
                unsafe { builder.launch(cfg_post) }.map_err(driver_err)?;
            }
        }

        buffer_pool.input = Some(d_input);
        buffer_pool.resized = Some(d_resized);

        Ok(trans_info)
    }

    /// Calculate resize parameters based on resize mode (using CudaImageProcessContext).
    fn calculate_resize_params_ctx(ctx: &CudaImageProcessContext) -> (i32, i32, i32, i32, f32) {
        let in_w = ctx.in_width as f32;
        let in_h = ctx.in_height as f32;
        let out_w = ctx.out_width() as f32;
        let out_h = ctx.out_height() as f32;

        match ctx.resize_mode() {
            ResizeMode::FitExact { .. } => (
                ctx.out_width() as i32,
                ctx.out_height() as i32,
                0,
                0,
                out_w / in_w,
            ),
            ResizeMode::Letterbox { .. } => {
                let scale = (out_w / in_w).min(out_h / in_h);
                let resized_w = (in_w * scale).round() as i32;
                let resized_h = (in_h * scale).round() as i32;
                let pad_left = (ctx.out_width() as i32 - resized_w) / 2;
                let pad_top = (ctx.out_height() as i32 - resized_h) / 2;
                (resized_w, resized_h, pad_left, pad_top, scale)
            }
            ResizeMode::FitAdaptive { .. } => {
                let scale = (out_w / in_w).min(out_h / in_h);
                let resized_w = (in_w * scale).round() as i32;
                let resized_h = (in_h * scale).round() as i32;
                (resized_w, resized_h, 0, 0, scale)
            }
            ResizeMode::FitWidth { .. } => {
                let scale = out_w / in_w;
                let resized_h = (in_h * scale).round() as i32;
                (ctx.out_width() as i32, resized_h, 0, 0, scale)
            }
            ResizeMode::FitHeight { .. } => {
                let scale = out_h / in_h;
                let resized_w = (in_w * scale).round() as i32;
                (resized_w, ctx.out_height() as i32, 0, 0, scale)
            }
        }
    }

    /// Preprocess a batch of images using ImagePlan.
    pub fn preprocess_batch_with_plan(
        &self,
        images: &[(&[u8], u32, u32)],
        plan: &ImagePlan,
    ) -> Result<(Vec<f32>, Vec<ImageTransformInfo>)> {
        if images.is_empty() {
            return Ok((vec![], vec![]));
        }

        let stream = self.ctx.default_stream();
        let mut buffer_pool = self
            .buffer_pool
            .lock()
            .map_err(|_| anyhow::anyhow!("Device buffer pool lock poisoned"))?;

        // Extract width and height from transforms
        let (out_width, out_height) = plan
            .transforms
            .iter()
            .find_map(|t| {
                if let crate::ImageTransform::Resize(mode) = t {
                    Some((mode.width(), mode.height()))
                } else {
                    None
                }
            })
            .unwrap_or((640, 640));
        let output_size_per_image = (out_height * out_width * 3) as usize;

        let output_len = output_size_per_image * images.len();
        let mut d_output = buffer_pool.take_or_alloc_output(&stream, output_len)?;

        // Upload mean/std if standardization is enabled
        let (d_mean, d_std) = match (plan.mean.as_ref(), plan.std.as_ref()) {
            (Some(mean), Some(std)) => {
                let mut mean_buf = buffer_pool.take_or_alloc_mean(&stream)?;
                stream
                    .memcpy_htod(mean, &mut mean_buf)
                    .map_err(driver_err)?;
                let mut std_buf = buffer_pool.take_or_alloc_std(&stream)?;
                stream.memcpy_htod(std, &mut std_buf).map_err(driver_err)?;
                (Some(mean_buf), Some(std_buf))
            }
            _ => (None, None),
        };

        let mut all_trans_info = Vec::with_capacity(images.len());
        for (idx, (data, width, height)) in images.iter().enumerate() {
            let ctx = CudaImageProcessContext::new(plan, *width, *height);

            let mut out_view = d_output
                .try_slice_mut(idx * output_size_per_image..(idx + 1) * output_size_per_image)
                .ok_or_else(|| anyhow::anyhow!("Failed to create CUDA output sub-slice"))?;

            let info = self.preprocess_image_with_context(
                &stream,
                data,
                &ctx,
                &mut out_view,
                d_mean.as_ref(),
                d_std.as_ref(),
                &mut buffer_pool,
            )?;
            all_trans_info.push(info);
        }

        stream.synchronize().map_err(driver_err)?;
        let mut output = vec![0.0f32; output_size_per_image * images.len()];
        stream
            .memcpy_dtoh(&d_output, &mut output)
            .map_err(driver_err)?;

        buffer_pool.output = Some(d_output);
        buffer_pool.mean = d_mean;
        buffer_pool.std = d_std;

        Ok((output, all_trans_info))
    }

    /// Preprocess a batch of images and return device buffer (zero-copy).
    ///
    /// This method keeps preprocessed data on the GPU, avoiding device-to-host copy.
    /// Use this for CUDA→CUDA workflows to maximize performance.
    ///
    /// Returns: (device_buffer, shape, transform_info)
    pub fn preprocess_batch_device(
        &self,
        images: &[(&[u8], u32, u32)],
        plan: &ImagePlan,
    ) -> Result<(CudaSlice<f32>, Vec<i64>, Vec<ImageTransformInfo>)> {
        if images.is_empty() {
            anyhow::bail!("No images provided for preprocessing");
        }

        let stream = self.ctx.default_stream();
        let mut buffer_pool = self
            .buffer_pool
            .lock()
            .map_err(|_| anyhow::anyhow!("Device buffer pool lock poisoned"))?;

        // Extract width and height from transforms
        let (out_width, out_height) = plan
            .transforms
            .iter()
            .find_map(|t| {
                if let crate::ImageTransform::Resize(mode) = t {
                    Some((mode.width(), mode.height()))
                } else {
                    None
                }
            })
            .unwrap_or((640, 640));
        let output_size_per_image = (out_height * out_width * 3) as usize;

        let output_len = output_size_per_image * images.len();
        let mut d_output = buffer_pool.take_or_alloc_output(&stream, output_len)?;

        // Upload mean/std if standardization is enabled
        let (d_mean, d_std) = match (plan.mean.as_ref(), plan.std.as_ref()) {
            (Some(mean), Some(std)) => {
                let mut mean_buf = buffer_pool.take_or_alloc_mean(&stream)?;
                stream
                    .memcpy_htod(mean, &mut mean_buf)
                    .map_err(driver_err)?;
                let mut std_buf = buffer_pool.take_or_alloc_std(&stream)?;
                stream.memcpy_htod(std, &mut std_buf).map_err(driver_err)?;
                (Some(mean_buf), Some(std_buf))
            }
            _ => (None, None),
        };

        let mut all_trans_info = Vec::with_capacity(images.len());
        for (idx, (data, width, height)) in images.iter().enumerate() {
            let ctx = CudaImageProcessContext::new(plan, *width, *height);

            let mut out_view = d_output
                .try_slice_mut(idx * output_size_per_image..(idx + 1) * output_size_per_image)
                .ok_or_else(|| anyhow::anyhow!("Failed to create CUDA output sub-slice"))?;

            let info = self.preprocess_image_with_context(
                &stream,
                data,
                &ctx,
                &mut out_view,
                d_mean.as_ref(),
                d_std.as_ref(),
                &mut buffer_pool,
            )?;
            all_trans_info.push(info);
        }

        // Synchronize before returning device pointer
        stream.synchronize().map_err(driver_err)?;

        // Return ownership of device buffer and cleanup other buffers
        buffer_pool.mean = d_mean;
        buffer_pool.std = d_std;

        // Build shape based on layout
        let shape = if images.len() == 1 {
            match plan.layout {
                ImageTensorLayout::NCHW => vec![1, 3, out_height as i64, out_width as i64],
                ImageTensorLayout::NHWC => vec![1, out_height as i64, out_width as i64, 3],
                ImageTensorLayout::CHW => vec![3, out_height as i64, out_width as i64],
                ImageTensorLayout::HWC => vec![out_height as i64, out_width as i64, 3],
            }
        } else {
            match plan.layout {
                ImageTensorLayout::NCHW | ImageTensorLayout::CHW => {
                    vec![images.len() as i64, 3, out_height as i64, out_width as i64]
                }
                ImageTensorLayout::NHWC | ImageTensorLayout::HWC => {
                    vec![images.len() as i64, out_height as i64, out_width as i64, 3]
                }
            }
        };

        Ok((d_output, shape, all_trans_info))
    }

    /// Get the CUDA stream.
    pub fn stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.ctx.default_stream()
    }

    /// Get device ID.
    pub fn device_id(&self) -> usize {
        self.ctx.cu_device() as usize
    }

    /// Execute transforms sequentially on CUDA device (for single image)
    /// This implements Pad/Crop/Resize operations using CUDA kernels
    fn execute_transforms_cuda(
        &self,
        image: &crate::Image,
        plan: &ImagePlan,
    ) -> Result<(crate::XAny, Vec<crate::ImageTransformInfo>)> {
        use cudarc::driver::LaunchConfig;

        let stream = self.stream();
        let mut buffer_pool = self
            .buffer_pool
            .lock()
            .map_err(|_| anyhow::anyhow!("Device buffer pool lock poisoned"))?;
        let (mut current_width, mut current_height) = image.dimensions();

        // Upload initial image to device
        let image_data = image.as_raw();
        let channels = 3i32;
        let current_size = (current_height * current_width * 3) as usize;

        let mut d_current = unsafe { stream.alloc::<u8>(current_size).map_err(driver_err)? };
        stream
            .memcpy_htod(image_data, &mut d_current)
            .map_err(driver_err)?;

        let (out_width, out_height) = plan
            .transforms
            .iter()
            .find_map(|t| {
                if let crate::ImageTransform::Resize(mode) = t {
                    Some((mode.width(), mode.height()))
                } else {
                    None
                }
            })
            .unwrap_or((640, 640));

        let mut merged_info = crate::ImageTransformInfo::default()
            .with_width_src(current_width)
            .with_height_src(current_height)
            .with_width_dst(out_width)
            .with_height_dst(out_height)
            .with_width_scale(1.0)
            .with_height_scale(1.0);

        // Execute each transform sequentially
        for (transform_idx, transform) in plan.transforms.iter().enumerate() {
            match transform {
                crate::ImageTransform::Pad(crate::PadMode::ToMultiple {
                    window_size,
                    fill_mode,
                }) => {
                    let (w_old, h_old) = (current_width as usize, current_height as usize);
                    let h_pad_total = (h_old / window_size + 1) * window_size - h_old;
                    let w_pad_total = (w_old / window_size + 1) * window_size - w_old;
                    let out_width = (w_old + w_pad_total) as u32;
                    let out_height = (h_old + h_pad_total) as u32;

                    let output_size = (out_height * out_width * 3) as usize;
                    let mut d_output =
                        unsafe { stream.alloc::<u8>(output_size).map_err(driver_err)? };

                    let cfg = LaunchConfig {
                        grid_dim: (out_width.div_ceil(16), out_height.div_ceil(16), 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };

                    let in_h = current_height as i32;
                    let in_w = current_width as i32;
                    let out_h = out_height as i32;
                    let out_w = out_width as i32;

                    match fill_mode {
                        crate::PadFillMode::Constant(value) => {
                            let mut builder = stream.launch_builder(&self.kernel_pad_constant);
                            unsafe {
                                builder
                                    .arg(&d_current)
                                    .arg(&mut d_output)
                                    .arg(&in_h)
                                    .arg(&in_w)
                                    .arg(&out_h)
                                    .arg(&out_w)
                                    .arg(&channels)
                                    .arg(value)
                                    .launch(cfg)
                            }
                            .map_err(driver_err)?;
                        }
                        crate::PadFillMode::Reflect => {
                            let mut builder = stream.launch_builder(&self.kernel_pad_reflect);
                            unsafe {
                                builder
                                    .arg(&d_current)
                                    .arg(&mut d_output)
                                    .arg(&in_h)
                                    .arg(&in_w)
                                    .arg(&out_h)
                                    .arg(&out_w)
                                    .arg(&channels)
                                    .launch(cfg)
                            }
                            .map_err(driver_err)?;
                        }
                        crate::PadFillMode::Replicate => {
                            let mut builder = stream.launch_builder(&self.kernel_pad_replicate);
                            unsafe {
                                builder
                                    .arg(&d_current)
                                    .arg(&mut d_output)
                                    .arg(&in_h)
                                    .arg(&in_w)
                                    .arg(&out_h)
                                    .arg(&out_w)
                                    .arg(&channels)
                                    .launch(cfg)
                            }
                            .map_err(driver_err)?;
                        }
                        crate::PadFillMode::Wrap => {
                            let mut builder = stream.launch_builder(&self.kernel_pad_wrap);
                            unsafe {
                                builder
                                    .arg(&d_current)
                                    .arg(&mut d_output)
                                    .arg(&in_h)
                                    .arg(&in_w)
                                    .arg(&out_h)
                                    .arg(&out_w)
                                    .arg(&channels)
                                    .launch(cfg)
                            }
                            .map_err(driver_err)?;
                        }
                    }

                    d_current = d_output;
                    current_width = out_width;
                    current_height = out_height;
                    // current_size = output_size;

                    let info = crate::ImageTransformInfo::default()
                        .with_width_src(w_old as u32)
                        .with_height_src(h_old as u32)
                        .with_width_dst(out_width)
                        .with_height_dst(out_height)
                        .with_width_pad(w_pad_total as f32)
                        .with_height_pad(h_pad_total as f32);
                    merged_info = merged_info.merge(&info);
                }
                crate::ImageTransform::Pad(crate::PadMode::Fixed {
                    top,
                    bottom,
                    left,
                    right,
                    fill_mode,
                }) => {
                    let (w_old, h_old) = (current_width, current_height);
                    let out_width = w_old + *left + *right;
                    let out_height = h_old + *top + *bottom;

                    let output_size = (out_height * out_width * 3) as usize;
                    let mut d_output =
                        unsafe { stream.alloc::<u8>(output_size).map_err(driver_err)? };

                    let cfg = LaunchConfig {
                        grid_dim: (out_width.div_ceil(16), out_height.div_ceil(16), 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };

                    let in_h = current_height as i32;
                    let in_w = current_width as i32;
                    let out_h = out_height as i32;
                    let out_w = out_width as i32;
                    let pad_top = *top as i32;
                    let pad_left = *left as i32;

                    match fill_mode {
                        crate::PadFillMode::Constant(value) => {
                            let mut builder =
                                stream.launch_builder(&self.kernel_pad_fixed_constant);
                            unsafe {
                                builder
                                    .arg(&d_current)
                                    .arg(&mut d_output)
                                    .arg(&in_h)
                                    .arg(&in_w)
                                    .arg(&out_h)
                                    .arg(&out_w)
                                    .arg(&channels)
                                    .arg(&pad_top)
                                    .arg(&pad_left)
                                    .arg(value)
                                    .launch(cfg)
                            }
                            .map_err(driver_err)?;
                        }
                        crate::PadFillMode::Reflect => {
                            let mut builder = stream.launch_builder(&self.kernel_pad_fixed_reflect);
                            unsafe {
                                builder
                                    .arg(&d_current)
                                    .arg(&mut d_output)
                                    .arg(&in_h)
                                    .arg(&in_w)
                                    .arg(&out_h)
                                    .arg(&out_w)
                                    .arg(&channels)
                                    .arg(&pad_top)
                                    .arg(&pad_left)
                                    .launch(cfg)
                            }
                            .map_err(driver_err)?;
                        }
                        crate::PadFillMode::Replicate => {
                            let mut builder =
                                stream.launch_builder(&self.kernel_pad_fixed_replicate);
                            unsafe {
                                builder
                                    .arg(&d_current)
                                    .arg(&mut d_output)
                                    .arg(&in_h)
                                    .arg(&in_w)
                                    .arg(&out_h)
                                    .arg(&out_w)
                                    .arg(&channels)
                                    .arg(&pad_top)
                                    .arg(&pad_left)
                                    .launch(cfg)
                            }
                            .map_err(driver_err)?;
                        }
                        crate::PadFillMode::Wrap => {
                            let mut builder = stream.launch_builder(&self.kernel_pad_fixed_wrap);
                            unsafe {
                                builder
                                    .arg(&d_current)
                                    .arg(&mut d_output)
                                    .arg(&in_h)
                                    .arg(&in_w)
                                    .arg(&out_h)
                                    .arg(&out_w)
                                    .arg(&channels)
                                    .arg(&pad_top)
                                    .arg(&pad_left)
                                    .launch(cfg)
                            }
                            .map_err(driver_err)?;
                        }
                    }

                    d_current = d_output;
                    current_width = out_width;
                    current_height = out_height;

                    let info = crate::ImageTransformInfo::default()
                        .with_width_src(w_old)
                        .with_height_src(h_old)
                        .with_width_dst(out_width)
                        .with_height_dst(out_height)
                        .with_width_pad((*left + *right) as f32)
                        .with_height_pad((*top + *bottom) as f32);
                    merged_info = merged_info.merge(&info);
                }
                crate::ImageTransform::Crop(crate::CropMode::Center { width, height }) => {
                    let crop_width = *width;
                    let crop_height = *height;

                    let (w_old, h_old) = (current_width, current_height);
                    let pad_left = if crop_width > w_old {
                        ((crop_width - w_old) / 2) as f32
                    } else {
                        0.0
                    };
                    let pad_top = if crop_height > h_old {
                        ((crop_height - h_old) / 2) as f32
                    } else {
                        0.0
                    };

                    let output_size = (crop_height * crop_width * 3) as usize;
                    let mut d_output =
                        unsafe { stream.alloc::<u8>(output_size).map_err(driver_err)? };

                    let cfg = LaunchConfig {
                        grid_dim: (crop_width.div_ceil(16), crop_height.div_ceil(16), 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };

                    let in_h = current_height as i32;
                    let in_w = current_width as i32;
                    let crop_h = crop_height as i32;
                    let crop_w = crop_width as i32;

                    let mut builder = stream.launch_builder(&self.kernel_crop_center);
                    unsafe {
                        builder
                            .arg(&d_current)
                            .arg(&mut d_output)
                            .arg(&in_h)
                            .arg(&in_w)
                            .arg(&crop_h)
                            .arg(&crop_w)
                            .arg(&channels)
                            .launch(cfg)
                    }
                    .map_err(driver_err)?;

                    d_current = d_output;
                    current_width = crop_width;
                    current_height = crop_height;
                    // current_size = output_size;

                    let info = crate::ImageTransformInfo::default()
                        .with_width_src(w_old)
                        .with_height_src(h_old)
                        .with_width_dst(crop_width)
                        .with_height_dst(crop_height)
                        .with_width_scale(1.0)
                        .with_height_scale(1.0)
                        .with_width_pad(pad_left)
                        .with_height_pad(pad_top);
                    merged_info = merged_info.merge(&info);
                }
                crate::ImageTransform::Resize(_) => {
                    if transform_idx + 1 != plan.transforms.len() {
                        anyhow::bail!(
                            "CUDA sequential transform execution requires Resize to be the last transform"
                        );
                    }

                    let ctx = CudaImageProcessContext::new(plan, current_width, current_height);

                    let output_len = (ctx.out_width() * ctx.out_height() * 3) as usize;
                    let mut d_output_f32 =
                        stream.alloc_zeros::<f32>(output_len).map_err(driver_err)?;
                    let mut out_view = d_output_f32
                        .try_slice_mut(0..output_len)
                        .ok_or_else(|| anyhow::anyhow!("Failed to create CUDA output sub-slice"))?;

                    let (d_mean, d_std) = match (plan.mean.as_ref(), plan.std.as_ref()) {
                        (Some(mean), Some(std)) => {
                            let mut mean_buf = buffer_pool.take_or_alloc_mean(&stream)?;
                            stream
                                .memcpy_htod(mean, &mut mean_buf)
                                .map_err(driver_err)?;
                            let mut std_buf = buffer_pool.take_or_alloc_std(&stream)?;
                            stream.memcpy_htod(std, &mut std_buf).map_err(driver_err)?;
                            (Some(mean_buf), Some(std_buf))
                        }
                        _ => (None, None),
                    };

                    let resize_info = self.preprocess_device_u8_with_context(
                        &stream,
                        &d_current,
                        &ctx,
                        &mut out_view,
                        d_mean.as_ref(),
                        d_std.as_ref(),
                        &mut buffer_pool,
                    )?;

                    buffer_pool.mean = d_mean;
                    buffer_pool.std = d_std;

                    stream.synchronize().map_err(driver_err)?;

                    let shape = match plan.layout {
                        ImageTensorLayout::NCHW => {
                            vec![1, 3, ctx.out_height() as i64, ctx.out_width() as i64]
                        }
                        ImageTensorLayout::NHWC => {
                            vec![1, ctx.out_height() as i64, ctx.out_width() as i64, 3]
                        }
                        ImageTensorLayout::CHW => {
                            vec![3, ctx.out_height() as i64, ctx.out_width() as i64]
                        }
                        ImageTensorLayout::HWC => {
                            vec![ctx.out_height() as i64, ctx.out_width() as i64, 3]
                        }
                    };

                    merged_info = merged_info.merge(&resize_info);

                    let cuda_tensor =
                        crate::XCuda::new(d_output_f32, shape, self.stream(), self.device_id());
                    return Ok((crate::XAny::from_device(cuda_tensor), vec![merged_info]));
                }
                _ => anyhow::bail!("Unsupported transform in CUDA path"),
            }
        }

        // Convert final u8 buffer to f32 tensor with postprocessing
        let output_size_f32 = (current_height * current_width * 3) as usize;
        let mut d_output_f32 = unsafe { stream.alloc::<f32>(output_size_f32).map_err(driver_err)? };

        let cfg = LaunchConfig {
            grid_dim: (current_width.div_ceil(16), current_height.div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let normalize_scale = if plan.normalize {
            1.0f32 / 255.0f32
        } else {
            1.0f32
        };
        let apply_unsigned = if plan.unsigned { 1i32 } else { 0i32 };

        let (d_mean, d_std) = match (plan.mean.as_ref(), plan.std.as_ref()) {
            (Some(mean), Some(std)) => {
                let mut mean_buf = buffer_pool.take_or_alloc_mean(&stream)?;
                stream
                    .memcpy_htod(mean, &mut mean_buf)
                    .map_err(driver_err)?;
                let mut std_buf = buffer_pool.take_or_alloc_std(&stream)?;
                stream.memcpy_htod(std, &mut std_buf).map_err(driver_err)?;
                (Some(mean_buf), Some(std_buf))
            }
            _ => (None, None),
        };
        let null_ptr: u64 = 0;
        let h_i32 = current_height as i32;
        let w_i32 = current_width as i32;
        let padding_value = 0u8;

        let kernel = if plan.layout.is_channels_first() {
            &self.kernel_post_u8_to_nchw_f32
        } else {
            &self.kernel_post_u8_to_nhwc_f32
        };

        let mut builder = stream.launch_builder(kernel);
        builder
            .arg(&d_current)
            .arg(&mut d_output_f32)
            .arg(&h_i32)
            .arg(&w_i32)
            .arg(&h_i32)
            .arg(&w_i32)
            .arg(&0i32)
            .arg(&0i32) // pad_top, pad_left
            .arg(&padding_value)
            .arg(&normalize_scale);

        if let Some(mean) = d_mean.as_ref() {
            builder.arg(mean);
        } else {
            builder.arg(&null_ptr);
        }
        if let Some(std) = d_std.as_ref() {
            builder.arg(std);
        } else {
            builder.arg(&null_ptr);
        }
        builder.arg(&apply_unsigned);

        unsafe { builder.launch(cfg) }.map_err(driver_err)?;
        stream.synchronize().map_err(driver_err)?;

        buffer_pool.mean = d_mean;
        buffer_pool.std = d_std;

        let shape = match plan.layout {
            ImageTensorLayout::NCHW => vec![1, 3, current_height as i64, current_width as i64],
            ImageTensorLayout::NHWC => vec![1, current_height as i64, current_width as i64, 3],
            ImageTensorLayout::CHW => vec![3, current_height as i64, current_width as i64],
            ImageTensorLayout::HWC => vec![current_height as i64, current_width as i64, 3],
        };

        let cuda_tensor = crate::XCuda::new(d_output_f32, shape, self.stream(), self.device_id());
        Ok((crate::XAny::from_device(cuda_tensor), vec![merged_info]))
    }

    #[allow(clippy::too_many_arguments)]
    fn preprocess_device_u8_with_context(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
        d_input: &CudaSlice<u8>,
        ctx: &CudaImageProcessContext,
        d_output: &mut cudarc::driver::CudaViewMut<f32>,
        d_mean: Option<&cudarc::driver::CudaSlice<f32>>,
        d_std: Option<&cudarc::driver::CudaSlice<f32>>,
        buffer_pool: &mut DeviceBufferPool,
    ) -> Result<ImageTransformInfo> {
        let (resized_w, resized_h, pad_left, pad_top, scale) =
            Self::calculate_resize_params_ctx(ctx);

        let resized_w_u32 = resized_w.max(0) as u32;
        let resized_h_u32 = resized_h.max(0) as u32;

        let (scale_w, scale_h) = match ctx.resize_mode() {
            ResizeMode::FitExact { .. } => (
                ctx.out_width() as f32 / ctx.in_width as f32,
                ctx.out_height() as f32 / ctx.in_height as f32,
            ),
            ResizeMode::FitWidth { .. } => (1.0, scale),
            ResizeMode::FitHeight { .. } => (scale, 1.0),
            ResizeMode::FitAdaptive { .. } | ResizeMode::Letterbox { .. } => (scale, scale),
        };

        let trans_info = ImageTransformInfo::default()
            .with_width_src(ctx.in_width)
            .with_height_src(ctx.in_height)
            .with_width_dst(ctx.out_width())
            .with_height_dst(ctx.out_height())
            .with_width_scale(scale_w)
            .with_height_scale(scale_h)
            .with_width_pad(pad_left as f32)
            .with_height_pad(pad_top as f32);

        if resized_w_u32 == 0 || resized_h_u32 == 0 {
            anyhow::bail!("Invalid resized size: {resized_w}x{resized_h}");
        }

        let block_x = 16u32;
        let block_y = 16u32;
        let in_h_i32 = ctx.in_height as i32;
        let in_w_i32 = ctx.in_width as i32;
        let resized_h_i32 = resized_h_u32 as i32;
        let resized_w_i32 = resized_w_u32 as i32;

        let resized_size = (resized_w_u32 * resized_h_u32 * 3) as usize;
        let mut d_resized = buffer_pool.take_or_alloc_resized(stream, resized_size)?;

        match ctx.resize_alg() {
            ResizeAlg::Nearest => {
                let cfg = LaunchConfig {
                    grid_dim: (
                        resized_w_u32.div_ceil(block_x),
                        resized_h_u32.div_ceil(block_y),
                        1,
                    ),
                    block_dim: (block_x, block_y, 1),
                    shared_mem_bytes: 0,
                };

                unsafe {
                    stream
                        .launch_builder(&self.kernel_resize_nearest_u8x3)
                        .arg(d_input)
                        .arg(&mut d_resized)
                        .arg(&in_h_i32)
                        .arg(&in_w_i32)
                        .arg(&resized_h_i32)
                        .arg(&resized_w_i32)
                        .launch(cfg)
                }
                .map_err(driver_err)?;
            }
            ResizeAlg::Convolution(filter) | ResizeAlg::Interpolation(filter) => {
                let adaptive_kernel_size = matches!(ctx.resize_alg(), ResizeAlg::Convolution(_));

                let vert = self.get_or_create_vert_coeffs(
                    stream,
                    ConvKey {
                        in_size: ctx.in_height,
                        out_size: resized_h_u32,
                        filter,
                        adaptive_kernel_size,
                    },
                )?;
                let horiz = self.get_or_create_horiz_coeffs(
                    stream,
                    ConvKey {
                        in_size: ctx.in_width,
                        out_size: resized_w_u32,
                        filter,
                        adaptive_kernel_size,
                    },
                )?;

                let temp_width = horiz.temp_width;
                let offset_x_i32 = horiz.x_first;
                let v_precision: i32 = vert.precision;
                let h_precision: i32 = horiz.coeffs.precision;

                let temp_size = (temp_width * resized_h_u32 * 3) as usize;
                let mut d_temp = buffer_pool.take_or_alloc_temp(stream, temp_size)?;

                let cfg_v = LaunchConfig {
                    grid_dim: (
                        temp_width.div_ceil(block_x),
                        resized_h_u32.div_ceil(block_y),
                        1,
                    ),
                    block_dim: (block_x, block_y, 1),
                    shared_mem_bytes: 0,
                };
                let out_h_i32 = resized_h_u32 as i32;
                let out_w_i32 = temp_width as i32;

                unsafe {
                    stream
                        .launch_builder(&self.kernel_conv_vert_u8x3)
                        .arg(d_input)
                        .arg(&mut d_temp)
                        .arg(&in_h_i32)
                        .arg(&in_w_i32)
                        .arg(&out_h_i32)
                        .arg(&out_w_i32)
                        .arg(&offset_x_i32)
                        .arg(&*vert.d_starts)
                        .arg(&*vert.d_sizes)
                        .arg(&*vert.d_offsets)
                        .arg(&*vert.d_coeffs)
                        .arg(&v_precision)
                        .launch(cfg_v)
                }
                .map_err(driver_err)?;

                let cfg_h = LaunchConfig {
                    grid_dim: (
                        resized_w_u32.div_ceil(block_x),
                        resized_h_u32.div_ceil(block_y),
                        1,
                    ),
                    block_dim: (block_x, block_y, 1),
                    shared_mem_bytes: 0,
                };
                let height_i32 = resized_h_u32 as i32;
                let in_w_h_i32 = temp_width as i32;
                let out_w_h_i32 = resized_w_u32 as i32;

                unsafe {
                    stream
                        .launch_builder(&self.kernel_conv_horiz_u8x3)
                        .arg(&d_temp)
                        .arg(&mut d_resized)
                        .arg(&height_i32)
                        .arg(&in_w_h_i32)
                        .arg(&out_w_h_i32)
                        .arg(&*horiz.coeffs.d_starts)
                        .arg(&*horiz.coeffs.d_sizes)
                        .arg(&*horiz.coeffs.d_offsets)
                        .arg(&*horiz.coeffs.d_coeffs)
                        .arg(&h_precision)
                        .launch(cfg_h)
                }
                .map_err(driver_err)?;

                buffer_pool.temp = Some(d_temp);
            }
            ResizeAlg::SuperSampling(_, _) => {
                anyhow::bail!("CUDA backend does not support SuperSampling")
            }
        }

        let normalize_scale = if ctx.normalize() {
            1.0f32 / 255.0f32
        } else {
            1.0f32
        };
        let apply_unsigned: i32 = if ctx.unsigned() { 1 } else { 0 };
        let pad_top_i32 = pad_top;
        let pad_left_i32 = pad_left;
        let padding_value_u8 = ctx.padding_value();
        let out_h_full_i32 = ctx.out_height() as i32;
        let out_w_full_i32 = ctx.out_width() as i32;
        let resized_h_i32 = resized_h_u32 as i32;
        let resized_w_i32 = resized_w_u32 as i32;
        let null_ptr: u64 = 0;
        let cfg_post = LaunchConfig {
            grid_dim: (
                ctx.out_width().div_ceil(block_x),
                ctx.out_height().div_ceil(block_y),
                1,
            ),
            block_dim: (block_x, block_y, 1),
            shared_mem_bytes: 0,
        };

        match ctx.layout() {
            ImageTensorLayout::NCHW | ImageTensorLayout::CHW => {
                let mut builder = stream.launch_builder(&self.kernel_post_u8_to_nchw_f32);
                builder
                    .arg(&d_resized)
                    .arg(d_output)
                    .arg(&resized_h_i32)
                    .arg(&resized_w_i32)
                    .arg(&out_h_full_i32)
                    .arg(&out_w_full_i32)
                    .arg(&pad_top_i32)
                    .arg(&pad_left_i32)
                    .arg(&padding_value_u8)
                    .arg(&normalize_scale);
                if let Some(mean) = d_mean {
                    builder.arg(mean);
                } else {
                    builder.arg(&null_ptr);
                }
                if let Some(std) = d_std {
                    builder.arg(std);
                } else {
                    builder.arg(&null_ptr);
                }
                builder.arg(&apply_unsigned);
                unsafe { builder.launch(cfg_post) }.map_err(driver_err)?;
            }
            ImageTensorLayout::NHWC | ImageTensorLayout::HWC => {
                let mut builder = stream.launch_builder(&self.kernel_post_u8_to_nhwc_f32);
                builder
                    .arg(&d_resized)
                    .arg(d_output)
                    .arg(&resized_h_i32)
                    .arg(&resized_w_i32)
                    .arg(&out_h_full_i32)
                    .arg(&out_w_full_i32)
                    .arg(&pad_top_i32)
                    .arg(&pad_left_i32)
                    .arg(&padding_value_u8)
                    .arg(&normalize_scale);
                if let Some(mean) = d_mean {
                    builder.arg(mean);
                } else {
                    builder.arg(&null_ptr);
                }
                if let Some(std) = d_std {
                    builder.arg(std);
                } else {
                    builder.arg(&null_ptr);
                }
                builder.arg(&apply_unsigned);
                unsafe { builder.launch(cfg_post) }.map_err(driver_err)?;
            }
        }

        buffer_pool.resized = Some(d_resized);

        Ok(trans_info)
    }
}
