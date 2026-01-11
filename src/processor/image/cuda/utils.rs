use anyhow::Result;
use cudarc::driver::CudaSlice;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use crate::ResizeFilter;

pub(crate) fn sys_err(e: cudarc::driver::sys::CUresult) -> anyhow::Error {
    anyhow::anyhow!("CUDA sys error: {:?}", e)
}

/// Convert cudarc DriverError to anyhow::Error
pub(crate) fn driver_err(e: cudarc::driver::result::DriverError) -> anyhow::Error {
    anyhow::anyhow!("CUDA driver error: {:?}", e)
}

pub(crate) struct PinnedHostBuffer {
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
pub(crate) struct ConvKey {
    pub(crate) in_size: u32,
    pub(crate) out_size: u32,
    pub(crate) filter: ResizeFilter,
    pub(crate) adaptive_kernel_size: bool,
}

pub(crate) struct DeviceConvCoeffs {
    pub(crate) d_starts: Arc<CudaSlice<i32>>,
    pub(crate) d_sizes: Arc<CudaSlice<i32>>,
    pub(crate) d_offsets: Arc<CudaSlice<i32>>,
    pub(crate) d_coeffs: Arc<CudaSlice<i16>>,
    pub(crate) precision: i32,
}

pub(crate) struct DeviceHorizConvCoeffs {
    pub(crate) coeffs: DeviceConvCoeffs,
    pub(crate) x_first: i32,
    pub(crate) temp_width: u32,
}

pub(crate) struct ConvCoeffCache {
    pub(crate) max_entries: usize,
    pub(crate) vert: HashMap<ConvKey, Arc<DeviceConvCoeffs>>,
    pub(crate) horiz: HashMap<ConvKey, Arc<DeviceHorizConvCoeffs>>,
}

#[derive(Default)]
pub(crate) struct DeviceBufferPool {
    pub(crate) input_len: usize,
    pub(crate) input: Option<CudaSlice<u8>>,
    pub(crate) pinned_input: PinnedHostBuffer,
    pub(crate) temp_len: usize,
    pub(crate) temp: Option<CudaSlice<u8>>,
    pub(crate) resized_len: usize,
    pub(crate) resized: Option<CudaSlice<u8>>,
    pub(crate) output_len: usize,
    pub(crate) output: Option<CudaSlice<f32>>,
    pub(crate) mean: Option<CudaSlice<f32>>,
    pub(crate) std: Option<CudaSlice<f32>>,
}

impl DeviceBufferPool {
    pub(crate) fn new() -> Self {
        Self {
            pinned_input: PinnedHostBuffer::new(),
            ..Default::default()
        }
    }

    pub(crate) fn fill_pinned_input(&mut self, src: &[u8]) -> Result<&[u8]> {
        self.pinned_input.fill_from(src)
    }

    pub(crate) fn take_or_alloc_input(
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

    pub(crate) fn take_or_alloc_temp(
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

    pub(crate) fn take_or_alloc_resized(
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

    pub(crate) fn take_or_alloc_output(
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

    pub(crate) fn take_or_alloc_mean(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<CudaSlice<f32>> {
        if let Some(buf) = self.mean.take() {
            return Ok(buf);
        }
        unsafe { stream.alloc::<f32>(3).map_err(driver_err) }
    }

    pub(crate) fn take_or_alloc_std(
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
    pub(crate) fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            vert: HashMap::new(),
            horiz: HashMap::new(),
        }
    }

    pub(crate) fn evict_if_needed(&mut self) {
        let total = self.vert.len() + self.horiz.len();
        if total >= self.max_entries {
            self.vert.clear();
            self.horiz.clear();
        }
    }
}
