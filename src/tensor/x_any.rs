use anyhow::Result;

#[cfg(feature = "cuda")]
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
    #[cfg(feature = "cuda")]
    Device(XCuda),
}

impl XAny {
    /// Create from host tensor.
    pub fn from_host(x: X) -> Self {
        XAny::Host(x)
    }

    /// Create from CUDA device tensor.
    #[cfg(feature = "cuda")]
    pub fn from_device(cuda_tensor: XCuda) -> Self {
        XAny::Device(cuda_tensor)
    }

    /// Check if tensor is on CUDA device.
    pub fn is_device(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            matches!(self, XAny::Device(_))
        }
        #[cfg(not(feature = "cuda"))]
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
            #[cfg(feature = "cuda")]
            XAny::Device(cuda_tensor) => cuda_tensor.to_host(),
        }
    }

    /// Try to get reference to host tensor (fails if on device).
    pub fn try_host_ref(&self) -> Option<&X> {
        match self {
            XAny::Host(x) => Some(x),
            #[cfg(feature = "cuda")]
            XAny::Device(_) => None,
        }
    }

    /// Try to get reference to device tensor (fails if on host).
    #[cfg(feature = "cuda")]
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
            #[cfg(feature = "cuda")]
            XAny::Device(cuda_tensor) => cuda_tensor.shape.iter().map(|&x| x as usize).collect(),
        }
    }
}

impl From<X> for XAny {
    fn from(x: X) -> Self {
        XAny::Host(x)
    }
}

#[cfg(feature = "cuda")]
impl From<XCuda> for XAny {
    fn from(cuda_tensor: XCuda) -> Self {
        XAny::Device(cuda_tensor)
    }
}
