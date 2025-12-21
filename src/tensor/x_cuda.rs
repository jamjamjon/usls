use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::sync::Arc;

use crate::X;

/// Device tensor holding CUDA memory without host copy.
///
/// This struct enables zero-copy CUDA→ORT inference by keeping
/// preprocessed data on the device until inference time.

#[derive(Debug)]
pub struct XCuda {
    /// CUDA device buffer
    pub buffer: CudaSlice<f32>,
    /// Tensor shape
    pub shape: Vec<i64>,
    /// CUDA stream for synchronization
    pub stream: Arc<CudaStream>,
    /// Device ID
    pub device_id: usize,
}

impl XCuda {
    pub fn new(
        buffer: CudaSlice<f32>,
        shape: Vec<i64>,
        stream: Arc<CudaStream>,
        device_id: usize,
    ) -> Self {
        Self {
            buffer,
            shape,
            stream,
            device_id,
        }
    }

    /// Get raw device pointer.
    #[inline]
    pub fn device_ptr(&self) -> *mut f32 {
        let (ptr, _guard) = self.buffer.device_ptr(&self.stream);
        ptr as *mut f32
    }

    /// Get shape as i64 slice.
    #[inline]
    pub fn shape_i64(&self) -> &[i64] {
        &self.shape
    }

    /// Synchronize CUDA stream.
    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("CUDA stream sync failed: {:?}", e))
    }

    /// Copy to host as X (fallback for non-CUDA models).
    pub fn to_host(&self) -> Result<X> {
        let mut output = vec![0.0f32; self.buffer.len()];
        self.stream
            .memcpy_dtoh(&self.buffer, &mut output)
            .map_err(|e| anyhow::anyhow!("CUDA D2H copy failed: {:?}", e))?;

        let shape: Vec<usize> = self.shape.iter().map(|&x| x as usize).collect();
        let arr = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), output)?;
        Ok(X(arr))
    }

    /// Get device ID.
    #[inline]
    pub fn device_id(&self) -> usize {
        self.device_id
    }
}
