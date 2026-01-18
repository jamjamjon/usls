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

    /// Get raw device pointer with stream synchronization.
    ///
    /// **Important**: This synchronizes the CUDA stream before returning the pointer,
    /// ensuring all pending operations on the buffer are complete. This is necessary
    /// when passing the pointer to external libraries (like ORT) that use different streams.
    #[inline]
    pub fn device_ptr(&self) -> *mut f32 {
        // Synchronize to ensure all cudarc operations are complete
        // before external code (ORT) accesses the buffer
        self.stream
            .synchronize()
            .expect("CUDA stream synchronization failed in device_ptr");
        // Get device pointer - the guard ensures sync but we already synced above
        let (ptr, _guard) = self.buffer.device_ptr(&self.stream);
        ptr as *mut f32
    }

    /// Get raw device pointer without synchronization.
    ///
    /// # Safety
    /// Caller must ensure the CUDA stream is synchronized before using the pointer
    /// with external libraries or different CUDA streams.
    #[inline]
    pub unsafe fn device_ptr_unsync(&self) -> *mut f32 {
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

    /// Insert a new axis at the given position (zero-copy, just reshape).
    ///
    /// This operation only modifies the shape metadata without copying data.
    /// For example, shape [N, C, H, W] with insert_axis(0) becomes [1, N, C, H, W].
    pub fn insert_axis(mut self, axis: usize) -> Result<Self> {
        if axis > self.shape.len() {
            anyhow::bail!(
                "insert_axis: axis {} out of bounds for shape {:?}",
                axis,
                self.shape
            );
        }
        self.shape.insert(axis, 1);
        Ok(self)
    }

    /// Reshape to a new shape (zero-copy, just update shape metadata).
    ///
    /// The total number of elements must remain the same.
    pub fn reshape(mut self, new_shape: Vec<i64>) -> Result<Self> {
        let old_size: i64 = self.shape.iter().product();
        let new_size: i64 = new_shape.iter().product();
        if old_size != new_size {
            anyhow::bail!(
                "reshape: cannot reshape from {:?} (size {}) to {:?} (size {})",
                self.shape,
                old_size,
                new_shape,
                new_size
            );
        }
        self.shape = new_shape;
        Ok(self)
    }

    /// Stack multiple CUDA tensors along a new axis.
    ///
    /// All tensors must have the same shape and be on the same device.
    /// This enables batch + multi-view processing.
    ///
    /// **Performance note**:
    /// - `axis=0`: Uses efficient D2D copy (zero host transfer)
    /// - `axis>0`: Falls back to host stack (requires D2H + H2D)
    ///
    /// # Example
    /// ```ignore
    /// // Stack [N, C, H, W] tensors into [B, N, C, H, W]
    /// let batch1 = processor.process(&views1)?;  // [3, C, H, W]
    /// let batch2 = processor.process(&views2)?;  // [3, C, H, W]
    /// let stacked = XCuda::stack(vec![batch1, batch2], 0)?;  // [2, 3, C, H, W]
    /// ```
    pub fn stack(tensors: Vec<Self>, axis: usize) -> Result<Self> {
        if tensors.is_empty() {
            anyhow::bail!("stack: cannot stack empty tensor list");
        }

        let first = &tensors[0];
        let base_shape = &first.shape;
        let stream = Arc::clone(&first.stream);
        let device_id = first.device_id;

        // Validate all tensors have same shape and device
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if &t.shape != base_shape {
                anyhow::bail!(
                    "stack: tensor {} has shape {:?}, expected {:?}",
                    i,
                    t.shape,
                    base_shape
                );
            }
            if t.device_id != device_id {
                anyhow::bail!(
                    "stack: tensor {} is on device {}, expected device {}",
                    i,
                    t.device_id,
                    device_id
                );
            }
        }

        if axis > base_shape.len() {
            anyhow::bail!(
                "stack: axis {} out of bounds for shape {:?}",
                axis,
                base_shape
            );
        }

        // For axis=0, we can use efficient D2D sequential copy
        // For axis>0, the memory layout is interleaved, requiring strided copy
        // Fall back to host-based stack for correctness
        if axis == 0 {
            // Calculate output shape: insert new dimension at axis 0
            let mut output_shape = base_shape.clone();
            output_shape.insert(0, tensors.len() as i64);

            // Allocate output buffer
            let elem_per_tensor = first.buffer.len();
            let total_elements = elem_per_tensor * tensors.len();
            let mut output_buffer = unsafe {
                stream
                    .alloc::<f32>(total_elements)
                    .map_err(|e| anyhow::anyhow!("stack: alloc failed: {:?}", e))?
            };

            // D2D copy each tensor sequentially
            for (i, tensor) in tensors.iter().enumerate() {
                let offset = i * elem_per_tensor;
                let mut dst_slice = output_buffer
                    .try_slice_mut(offset..offset + elem_per_tensor)
                    .ok_or_else(|| anyhow::anyhow!("stack: slice failed"))?;

                stream
                    .memcpy_dtod(&tensor.buffer, &mut dst_slice)
                    .map_err(|e| anyhow::anyhow!("stack: D2D copy failed: {:?}", e))?;
            }

            stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("stack: sync failed: {:?}", e))?;

            Ok(Self {
                buffer: output_buffer,
                shape: output_shape,
                stream,
                device_id,
            })
        } else {
            // For axis > 0, fall back to host-based stack for correct memory layout
            let host_tensors: Vec<X> = tensors
                .iter()
                .map(|t| t.to_host())
                .collect::<Result<Vec<_>>>()?;

            let views: Vec<_> = host_tensors.iter().map(|t| t.0.view()).collect();
            let stacked = ndarray::stack(ndarray::Axis(axis), &views)?;

            let output_shape: Vec<i64> = stacked.shape().iter().map(|&x| x as i64).collect();
            let total_elements = stacked.len();

            let mut output_buffer = unsafe {
                stream
                    .alloc::<f32>(total_elements)
                    .map_err(|e| anyhow::anyhow!("stack: alloc failed: {:?}", e))?
            };

            stream
                .memcpy_htod(
                    stacked
                        .as_slice()
                        .ok_or_else(|| anyhow::anyhow!("stack: failed to get contiguous slice"))?,
                    &mut output_buffer,
                )
                .map_err(|e| anyhow::anyhow!("stack: H2D copy failed: {:?}", e))?;

            Ok(Self {
                buffer: output_buffer,
                shape: output_shape,
                stream,
                device_id,
            })
        }
    }

    /// Concatenate multiple CUDA tensors along an existing axis (device-to-device copy).
    ///
    /// All tensors must have the same shape except for the concatenation axis.
    ///
    /// # Example
    /// ```ignore
    /// // Concat [N, C, H, W] tensors along axis 0
    /// let t1 = ...;  // [2, C, H, W]
    /// let t2 = ...;  // [3, C, H, W]
    /// let concatenated = XCuda::concat(&[t1, t2], 0)?;  // [5, C, H, W]
    /// ```
    pub fn concat(tensors: Vec<Self>, axis: usize) -> Result<Self> {
        if tensors.is_empty() {
            anyhow::bail!("concat: cannot concat empty tensor list");
        }

        let first = &tensors[0];
        let ndim = first.shape.len();
        let stream = Arc::clone(&first.stream);
        let device_id = first.device_id;

        if axis >= ndim {
            anyhow::bail!("concat: axis {} out of bounds for {}D tensor", axis, ndim);
        }

        // Validate shapes match except on concat axis
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.shape.len() != ndim {
                anyhow::bail!(
                    "concat: tensor {} has {} dims, expected {}",
                    i,
                    t.shape.len(),
                    ndim
                );
            }
            for (d, (s1, s2)) in first.shape.iter().zip(t.shape.iter()).enumerate() {
                if d != axis && s1 != s2 {
                    anyhow::bail!("concat: shape mismatch at dim {}: {} vs {}", d, s1, s2);
                }
            }
            if t.device_id != device_id {
                anyhow::bail!(
                    "concat: tensor {} is on device {}, expected {}",
                    i,
                    t.device_id,
                    device_id
                );
            }
        }

        // Calculate output shape
        let mut output_shape = first.shape.clone();
        output_shape[axis] = tensors.iter().map(|t| t.shape[axis]).sum();

        // Total elements
        let total_elements: usize = tensors.iter().map(|t| t.buffer.len()).sum();
        let mut output_buffer = unsafe {
            stream
                .alloc::<f32>(total_elements)
                .map_err(|e| anyhow::anyhow!("concat: failed to allocate: {:?}", e))?
        };

        // For axis=0 concat, simple sequential copy works
        // For other axes, we need strided copy (more complex)
        if axis == 0 {
            let mut offset = 0;
            for tensor in tensors.iter() {
                let len = tensor.buffer.len();
                let mut dst_slice = output_buffer
                    .try_slice_mut(offset..offset + len)
                    .ok_or_else(|| anyhow::anyhow!("concat: slice failed"))?;
                stream
                    .memcpy_dtod(&tensor.buffer, &mut dst_slice)
                    .map_err(|e| anyhow::anyhow!("concat: D2D copy failed: {:?}", e))?;
                offset += len;
            }
        } else {
            // For non-zero axis, fall back to host concat (complex strided copy)
            // This is rare for the multi-view use case (usually stack at axis 0)
            let host_tensors: Vec<X> = tensors
                .iter()
                .map(|t| t.to_host())
                .collect::<Result<Vec<_>>>()?;

            let views: Vec<_> = host_tensors.iter().map(|t| t.0.view()).collect();
            let concatenated = ndarray::concatenate(ndarray::Axis(axis), &views)?;

            stream
                .memcpy_htod(concatenated.as_slice().unwrap(), &mut output_buffer)
                .map_err(|e| anyhow::anyhow!("concat: H2D copy failed: {:?}", e))?;
        }

        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("concat: sync failed: {:?}", e))?;

        Ok(Self {
            buffer: output_buffer,
            shape: output_shape,
            stream,
            device_id,
        })
    }
}
