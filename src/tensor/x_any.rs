use anyhow::Result;

#[cfg(feature = "cuda-runtime")]
use crate::XCuda;
use crate::X;

/// Processed tensor that can be either on CPU or CUDA device.
///
/// This enum enables zero-copy workflows:
/// - `Host`: Standard CPU tensor (ndarray-backed)
/// - `Device`: CUDA device memory (pointer + metadata)
///
/// # Performance
/// When using CUDA preprocessing + CUDA inference, this avoids:
/// 1. GPU→CPU copy after preprocessing
/// 2. CPU→GPU copy before inference
///
/// # Example
/// ```ignore
/// let processed = processor.process(&images)?;
/// match processed {
///     XAny::Host(x) => { /* CPU path */ }
///     XAny::Device(cuda_tensor) => { /* Zero-copy CUDA path */ }
/// }
/// ```
#[derive(Debug)]
pub enum XAny {
    /// CPU tensor (host memory)
    Host(X),
    /// CUDA tensor (device memory) - zero-copy
    #[cfg(feature = "cuda-runtime")]
    Device(XCuda),
}

impl XAny {
    /// Create from host tensor.
    pub fn from_host(x: X) -> Self {
        XAny::Host(x)
    }

    /// Create from CUDA device tensor.
    #[cfg(feature = "cuda-runtime")]
    pub fn from_device(cuda_tensor: XCuda) -> Self {
        XAny::Device(cuda_tensor)
    }

    /// Check if tensor is on CUDA device.
    pub fn is_device(&self) -> bool {
        #[cfg(feature = "cuda-runtime")]
        {
            matches!(self, XAny::Device(_))
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            false
        }
    }

    /// Check if tensor is on host.
    pub fn is_host(&self) -> bool {
        matches!(self, XAny::Host(_))
    }

    /// Get as host tensor, copying from device if necessary.
    pub fn as_host(&self) -> Result<X> {
        match self {
            XAny::Host(x) => Ok(x.clone()), // TODO: clone?
            #[cfg(feature = "cuda-runtime")]
            XAny::Device(cuda_tensor) => cuda_tensor.to_host(),
        }
    }

    /// Try to get reference to host tensor (fails if on device).
    pub fn try_host_ref(&self) -> Option<&X> {
        match self {
            XAny::Host(x) => Some(x),
            #[cfg(feature = "cuda-runtime")]
            XAny::Device(_) => None,
        }
    }

    /// Try to get reference to device tensor (fails if on host).
    #[cfg(feature = "cuda-runtime")]
    pub fn try_device_ref(&self) -> Option<&XCuda> {
        match self {
            XAny::Device(cuda_tensor) => Some(cuda_tensor),
            XAny::Host(_) => None,
        }
    }

    /// Get tensor shape.
    pub fn shape(&self) -> Vec<usize> {
        match self {
            XAny::Host(x) => x.dims().to_vec(),
            #[cfg(feature = "cuda-runtime")]
            XAny::Device(cuda_tensor) => cuda_tensor.shape.iter().map(|&x| x as usize).collect(),
        }
    }

    /// Insert a new axis at the given position (zero-copy for device tensors).
    ///
    /// This enables multi-view tensor construction without host-device copies:
    /// - For Host tensors: delegates to X::insert_axis
    /// - For Device tensors: only modifies shape metadata (zero-copy)
    ///
    /// # Example
    /// ```ignore
    /// // Transform [N, C, H, W] to [1, N, C, H, W] for multi-view input
    /// let x = processor.process(&images)?;
    /// let x = x.insert_axis(0)?;  // Zero-copy on CUDA!
    /// ```
    pub fn insert_axis(self, axis: usize) -> Result<Self> {
        match self {
            XAny::Host(x) => Ok(XAny::Host(x.insert_axis(axis)?)),
            #[cfg(feature = "cuda-runtime")]
            XAny::Device(cuda_tensor) => Ok(XAny::Device(cuda_tensor.insert_axis(axis)?)),
        }
    }

    /// Reshape tensor to a new shape (zero-copy for device tensors).
    ///
    /// The total number of elements must remain the same.
    pub fn reshape(self, new_shape: &[usize]) -> Result<Self> {
        match self {
            XAny::Host(x) => {
                let arr = x.0.into_shape_with_order(ndarray::IxDyn(new_shape))?;
                Ok(XAny::Host(X(arr)))
            }
            #[cfg(feature = "cuda-runtime")]
            XAny::Device(cuda_tensor) => {
                let new_shape_i64: Vec<i64> = new_shape.iter().map(|&x| x as i64).collect();
                Ok(XAny::Device(cuda_tensor.reshape(new_shape_i64)?))
            }
        }
    }

    /// Stack multiple tensors along a new axis.
    ///
    /// For CUDA tensors, this performs device-to-device copy (no host transfer).
    /// For host tensors, this uses ndarray stack.
    ///
    /// All tensors must:
    /// - Have the same shape
    /// - Be on the same device type (all Host or all Device)
    ///
    /// # Example
    /// ```ignore
    /// // Batch + multi-view: stack preprocessed batches
    /// let batch1 = processor.process(&views1)?;  // [num_views, C, H, W]
    /// let batch2 = processor.process(&views2)?;  // [num_views, C, H, W]
    /// let stacked = XAny::stack(vec![batch1, batch2], 0)?;  // [batch, num_views, C, H, W]
    /// ```
    pub fn stack(tensors: Vec<Self>, axis: usize) -> Result<Self> {
        if tensors.is_empty() {
            anyhow::bail!("stack: cannot stack empty tensor list");
        }

        // Check if all tensors are on the same device type
        let first_is_device = tensors[0].is_device();
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.is_device() != first_is_device {
                anyhow::bail!(
                    "stack: mixed device types - tensor 0 is {}, tensor {} is {}",
                    if first_is_device { "Device" } else { "Host" },
                    i,
                    if t.is_device() { "Device" } else { "Host" }
                );
            }
        }

        #[cfg(feature = "cuda-runtime")]
        if first_is_device {
            // Extract CUDA tensors and use XCuda::stack
            let cuda_tensors: Vec<XCuda> = tensors
                .into_iter()
                .map(|t| match t {
                    XAny::Device(cuda) => cuda,
                    _ => unreachable!(),
                })
                .collect();
            return Ok(XAny::Device(XCuda::stack(cuda_tensors, axis)?));
        }

        // Host path: use ndarray stack
        let host_tensors: Vec<X> = tensors
            .into_iter()
            .map(|t| match t {
                XAny::Host(x) => x,
                #[cfg(feature = "cuda-runtime")]
                XAny::Device(_) => unreachable!(),
            })
            .collect();

        let views: Vec<_> = host_tensors.iter().map(|t| t.0.view()).collect();
        let stacked = ndarray::stack(ndarray::Axis(axis), &views)?;
        Ok(XAny::Host(X(stacked)))
    }

    /// Concatenate multiple tensors along an existing axis.
    ///
    /// For CUDA tensors with axis=0, this performs device-to-device copy.
    /// All tensors must have the same shape except on the concat axis.
    pub fn concat(tensors: Vec<Self>, axis: usize) -> Result<Self> {
        if tensors.is_empty() {
            anyhow::bail!("concat: cannot concat empty tensor list");
        }

        let first_is_device = tensors[0].is_device();
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.is_device() != first_is_device {
                anyhow::bail!("concat: mixed device types at tensor {i}");
            }
        }

        #[cfg(feature = "cuda-runtime")]
        if first_is_device {
            let cuda_tensors: Vec<XCuda> = tensors
                .into_iter()
                .map(|t| match t {
                    XAny::Device(cuda) => cuda,
                    _ => unreachable!(),
                })
                .collect();
            return Ok(XAny::Device(XCuda::concat(cuda_tensors, axis)?));
        }

        // Host path
        let host_tensors: Vec<X> = tensors
            .into_iter()
            .map(|t| match t {
                XAny::Host(x) => x,
                #[cfg(feature = "cuda-runtime")]
                XAny::Device(_) => unreachable!(),
            })
            .collect();

        let views: Vec<_> = host_tensors.iter().map(|t| t.0.view()).collect();
        let concatenated = ndarray::concatenate(ndarray::Axis(axis), &views)?;
        Ok(XAny::Host(X(concatenated)))
    }
}

impl From<X> for XAny {
    fn from(x: X) -> Self {
        XAny::Host(x)
    }
}

#[cfg(feature = "cuda-runtime")]
impl From<XCuda> for XAny {
    fn from(cuda_tensor: XCuda) -> Self {
        XAny::Device(cuda_tensor)
    }
}
