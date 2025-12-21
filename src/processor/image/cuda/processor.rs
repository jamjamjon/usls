//! CUDA context management for processing.
//!
//! This module provides GPU-accelerated image processing using cudarc.
//! Requires CUDA 11.x+ to be installed on the system.

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use crate::{
    compute_convolution_1d, CudaImageProcessContext, ImagePlan, ImageTensorLayout,
    ImageTransformInfo, ResizeFilter, ResizeMode,
};

fn sys_err(e: cudarc::driver::sys::CUresult) -> anyhow::Error {
    anyhow::anyhow!("CUDA sys error: {:?}", e)
}

struct PinnedHostBuffer {
    ptr: *mut u8,
    cap: usize,
}

unsafe impl Send for PinnedHostBuffer {}

impl PinnedHostBuffer {
    fn new() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            cap: 0,
        }
    }

    fn ensure_capacity(&mut self, cap: usize) -> Result<()> {
        if self.cap >= cap {
            return Ok(());
        }
        self.free();

        let mut p: *mut c_void = std::ptr::null_mut();
        let res = unsafe { cudarc::driver::sys::cuMemAllocHost_v2(&mut p, cap) };
        if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            return Err(sys_err(res));
        }
        self.ptr = p as *mut u8;
        self.cap = cap;
        Ok(())
    }

    fn fill_from(&mut self, src: &[u8]) -> Result<&[u8]> {
        self.ensure_capacity(src.len())?;
        if src.is_empty() {
            return Ok(&[]);
        }
        unsafe {
            let dst = std::slice::from_raw_parts_mut(self.ptr, src.len());
            dst.copy_from_slice(src);
            Ok(std::slice::from_raw_parts(self.ptr, src.len()))
        }
    }

    fn free(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let p = self.ptr as *mut c_void;
        let _ = unsafe { cudarc::driver::sys::cuMemFreeHost(p) };
        self.ptr = std::ptr::null_mut();
        self.cap = 0;
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        self.free();
    }
}

impl Default for PinnedHostBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConvKey {
    in_size: u32,
    out_size: u32,
    filter: ResizeFilter,
    adaptive_kernel_size: bool,
}

struct DeviceConvCoeffs {
    d_starts: Arc<CudaSlice<i32>>,
    d_sizes: Arc<CudaSlice<i32>>,
    d_offsets: Arc<CudaSlice<i32>>,
    d_coeffs: Arc<CudaSlice<i16>>,
    precision: i32,
}

struct DeviceHorizConvCoeffs {
    coeffs: DeviceConvCoeffs,
    x_first: i32,
    temp_width: u32,
}

struct ConvCoeffCache {
    max_entries: usize,
    vert: HashMap<ConvKey, Arc<DeviceConvCoeffs>>,
    horiz: HashMap<ConvKey, Arc<DeviceHorizConvCoeffs>>,
}

#[derive(Default)]
struct DeviceBufferPool {
    input_len: usize,
    input: Option<CudaSlice<u8>>,
    pinned_input: PinnedHostBuffer,
    temp_len: usize,
    temp: Option<CudaSlice<u8>>,
    resized_len: usize,
    resized: Option<CudaSlice<u8>>,
    output_len: usize,
    output: Option<CudaSlice<f32>>,
    mean: Option<CudaSlice<f32>>,
    std: Option<CudaSlice<f32>>,
}

impl DeviceBufferPool {
    fn new() -> Self {
        Self {
            pinned_input: PinnedHostBuffer::new(),
            ..Default::default()
        }
    }

    fn fill_pinned_input(&mut self, src: &[u8]) -> Result<&[u8]> {
        self.pinned_input.fill_from(src)
    }

    fn take_or_alloc_input(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        len: usize,
    ) -> Result<CudaSlice<u8>> {
        if self.input_len == len {
            if let Some(buf) = self.input.take() {
                return Ok(buf);
            }
        }
        self.input_len = len;
        self.input = None;
        unsafe { stream.alloc::<u8>(len).map_err(driver_err) }
    }

    fn take_or_alloc_temp(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        len: usize,
    ) -> Result<CudaSlice<u8>> {
        if self.temp_len == len {
            if let Some(buf) = self.temp.take() {
                return Ok(buf);
            }
        }
        self.temp_len = len;
        self.temp = None;
        stream.alloc_zeros::<u8>(len).map_err(driver_err)
    }

    fn take_or_alloc_resized(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        len: usize,
    ) -> Result<CudaSlice<u8>> {
        if self.resized_len == len {
            if let Some(buf) = self.resized.take() {
                return Ok(buf);
            }
        }
        self.resized_len = len;
        self.resized = None;
        stream.alloc_zeros::<u8>(len).map_err(driver_err)
    }

    fn take_or_alloc_output(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        len: usize,
    ) -> Result<CudaSlice<f32>> {
        if self.output_len == len {
            if let Some(buf) = self.output.take() {
                return Ok(buf);
            }
        }
        self.output_len = len;
        self.output = None;
        stream.alloc_zeros::<f32>(len).map_err(driver_err)
    }

    fn take_or_alloc_mean(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<CudaSlice<f32>> {
        if let Some(buf) = self.mean.take() {
            return Ok(buf);
        }
        unsafe { stream.alloc::<f32>(3).map_err(driver_err) }
    }

    fn take_or_alloc_std(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<CudaSlice<f32>> {
        if let Some(buf) = self.std.take() {
            return Ok(buf);
        }
        unsafe { stream.alloc::<f32>(3).map_err(driver_err) }
    }
}

impl ConvCoeffCache {
    fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            vert: HashMap::new(),
            horiz: HashMap::new(),
        }
    }

    fn evict_if_needed(&mut self) {
        let total = self.vert.len() + self.horiz.len();
        if total >= self.max_entries {
            self.vert.clear();
            self.horiz.clear();
        }
    }
}

/// Convert cudarc DriverError to anyhow::Error
fn driver_err(e: cudarc::driver::result::DriverError) -> anyhow::Error {
    anyhow::anyhow!("CUDA driver error: {:?}", e)
}

/// CUDA source code for processing kernels (embedded at compile time).
const PREPROCESS_CUDA_SRC: &str = include_str!("process.cu");

/// CUDA preprocessor context managing device, streams, and kernels.
pub struct CudaPreprocessor {
    ctx: Arc<CudaContext>,
    kernel_conv_vert_u8x3: CudaFunction,
    kernel_conv_horiz_u8x3: CudaFunction,
    kernel_post_u8_to_nchw_f32: CudaFunction,
    kernel_post_u8_to_nhwc_f32: CudaFunction,
    conv_cache: Mutex<ConvCoeffCache>,
    buffer_pool: Mutex<DeviceBufferPool>,
}

impl CudaPreprocessor {
    /// Create a new CUDA preprocessor context.
    pub fn new(device_id: usize) -> Result<Self> {
        let ctx = CudaContext::new(device_id).map_err(driver_err)?;

        // Compile CUDA source to PTX at runtime
        let ptx = compile_ptx(PREPROCESS_CUDA_SRC)
            .map_err(|e| anyhow::anyhow!("Failed to compile CUDA kernels: {:?}", e))?;

        // Load PTX module and get kernel functions
        let module = ctx.load_module(ptx).map_err(driver_err)?;

        let kernel_conv_vert_u8x3 = module.load_function("conv_vert_u8x3").map_err(driver_err)?;
        let kernel_conv_horiz_u8x3 = module
            .load_function("conv_horiz_u8x3")
            .map_err(driver_err)?;
        let kernel_post_u8_to_nchw_f32 = module
            .load_function("postprocess_pad_u8_to_nchw_f32")
            .map_err(driver_err)?;
        let kernel_post_u8_to_nhwc_f32 = module
            .load_function("postprocess_pad_u8_to_nhwc_f32")
            .map_err(driver_err)?;

        Ok(Self {
            ctx,
            kernel_conv_vert_u8x3,
            kernel_conv_horiz_u8x3,
            kernel_post_u8_to_nchw_f32,
            kernel_post_u8_to_nhwc_f32,
            conv_cache: Mutex::new(ConvCoeffCache::new(32)),
            buffer_pool: Mutex::new(DeviceBufferPool::new()),
        })
    }

    /// Get the underlying CUDA context.
    #[allow(dead_code)]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Check if kernels are loaded.
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

        let trans_info = ImageTransformInfo::default()
            .with_width_src(ctx.in_width)
            .with_height_src(ctx.in_height)
            .with_width_dst(ctx.out_width())
            .with_height_dst(ctx.out_height())
            .with_width_scale(scale)
            .with_height_scale(scale)
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

        let resize_filter = ctx.resize_filter();
        let adaptive_kernel_size = true;
        let vert = self.get_or_create_vert_coeffs(
            stream,
            ConvKey {
                in_size: ctx.in_height,
                out_size: resized_h_u32,
                filter: resize_filter,
                adaptive_kernel_size,
            },
        )?;
        let horiz = self.get_or_create_horiz_coeffs(
            stream,
            ConvKey {
                in_size: ctx.in_width,
                out_size: resized_w_u32,
                filter: resize_filter,
                adaptive_kernel_size,
            },
        )?;

        let temp_width = horiz.temp_width;
        let offset_x_i32 = horiz.x_first;
        let v_precision: i32 = vert.precision;
        let h_precision: i32 = horiz.coeffs.precision;

        // Allocate temp and resized buffers
        let temp_size = (temp_width * resized_h_u32 * 3) as usize;
        let resized_size = (resized_w_u32 * resized_h_u32 * 3) as usize;
        let mut d_temp = buffer_pool.take_or_alloc_temp(stream, temp_size)?;
        let mut d_resized = buffer_pool.take_or_alloc_resized(stream, resized_size)?;

        // Vertical pass
        let block_x = 16u32;
        let block_y = 16u32;
        let cfg_v = LaunchConfig {
            grid_dim: (
                temp_width.div_ceil(block_x),
                resized_h_u32.div_ceil(block_y),
                1,
            ),
            block_dim: (block_x, block_y, 1),
            shared_mem_bytes: 0,
        };
        let in_h_i32 = ctx.in_height as i32;
        let in_w_i32 = ctx.in_width as i32;
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

        // Horizontal pass
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
        buffer_pool.temp = Some(d_temp);
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
            ResizeMode::FitExact => (
                ctx.out_width() as i32,
                ctx.out_height() as i32,
                0,
                0,
                out_w / in_w,
            ),
            ResizeMode::Letterbox => {
                let scale = (out_w / in_w).min(out_h / in_h);
                let resized_w = (in_w * scale).round() as i32;
                let resized_h = (in_h * scale).round() as i32;
                let pad_left = (ctx.out_width() as i32 - resized_w) / 2;
                let pad_top = (ctx.out_height() as i32 - resized_h) / 2;
                (resized_w, resized_h, pad_left, pad_top, scale)
            }
            ResizeMode::FitAdaptive => {
                let scale = (out_w / in_w).min(out_h / in_h);
                let resized_w = (in_w * scale).round() as i32;
                let resized_h = (in_h * scale).round() as i32;
                (resized_w, resized_h, 0, 0, scale)
            }
            ResizeMode::FitWidth => {
                let scale = out_w / in_w;
                let resized_h = (in_h * scale).round() as i32;
                (ctx.out_width() as i32, resized_h, 0, 0, scale)
            }
            ResizeMode::FitHeight => {
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
        let output_size_per_image = (plan.height * plan.width * 3) as usize;

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
        let output_size_per_image = (plan.height * plan.width * 3) as usize;

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
                ImageTensorLayout::NCHW => vec![1, 3, plan.height as i64, plan.width as i64],
                ImageTensorLayout::NHWC => vec![1, plan.height as i64, plan.width as i64, 3],
                ImageTensorLayout::CHW => vec![3, plan.height as i64, plan.width as i64],
                ImageTensorLayout::HWC => vec![plan.height as i64, plan.width as i64, 3],
            }
        } else {
            match plan.layout {
                ImageTensorLayout::NCHW | ImageTensorLayout::CHW => {
                    vec![
                        images.len() as i64,
                        3,
                        plan.height as i64,
                        plan.width as i64,
                    ]
                }
                ImageTensorLayout::NHWC | ImageTensorLayout::HWC => {
                    vec![
                        images.len() as i64,
                        plan.height as i64,
                        plan.width as i64,
                        3,
                    ]
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
}

impl std::fmt::Debug for CudaPreprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaPreprocessor")
            .field("is_ready", &self.is_ready())
            .finish()
    }
}
