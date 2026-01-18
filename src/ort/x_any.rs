use anyhow::Result;
use std::convert::TryFrom;

use crate::XAny;

// Implementation for inputs! macro compatibility with zero-copy support
impl<'a> TryFrom<&'a XAny> for ort::session::SessionInputValue<'a> {
    type Error = anyhow::Error;

    fn try_from(tensor: &'a XAny) -> Result<Self> {
        match tensor {
            XAny::Host(x) => {
                // Try zero-copy for f32 standard layout
                if x.0.is_standard_layout() {
                    let tensor_ref = ort::value::TensorRef::from_array_view(x.0.view())
                        .map_err(|e| anyhow::anyhow!("Failed to create TensorRef: {e:?}"))?;
                    Ok(ort::session::SessionInputValue::from(tensor_ref))
                } else {
                    // Fallback: create owned tensor
                    let value = ort::value::Value::from_array(x.0.clone())
                        .map_err(|e| anyhow::anyhow!("Failed to create Value: {e:?}"))?;
                    Ok(ort::session::SessionInputValue::from(value))
                }
            }
            #[cfg(feature = "cuda-runtime")]
            XAny::Device(cuda_tensor) => {
                // Zero-copy CUDA path: create ORT CUDA tensor directly
                use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
                use ort::tensor::Shape;
                use ort::value::TensorRefMut;

                // Create MemoryInfo for CUDA device
                let mem_info = MemoryInfo::new(
                    AllocationDevice::CUDA,
                    cuda_tensor.device_id() as i32,
                    AllocatorType::Device,
                    MemoryType::Default,
                )?;

                // Create TensorRefMut from raw CUDA pointer (f32 type)
                let tensor_ref: TensorRefMut<'a, _> = unsafe {
                    TensorRefMut::<f32>::from_raw(
                        mem_info,
                        cuda_tensor.device_ptr() as *mut std::ffi::c_void,
                        Shape::from(cuda_tensor.shape_i64()),
                    )?
                };

                Ok(ort::session::SessionInputValue::from(tensor_ref))
            }
        }
    }
}

impl TryFrom<XAny> for ort::session::SessionInputValue<'static> {
    type Error = anyhow::Error;

    fn try_from(tensor: XAny) -> Result<Self> {
        match tensor {
            XAny::Host(x) => {
                let value = ort::value::Value::from_array(x.0)
                    .map_err(|e| anyhow::anyhow!("Failed to create Value: {e:?}"))?;
                Ok(ort::session::SessionInputValue::from(value))
            }
            #[cfg(feature = "cuda-runtime")]
            XAny::Device(_) => {
                anyhow::bail!(
                    "Cannot convert owned CUDA tensor into SessionInputValue<'static>. Pass it by reference (e.g. inputs![&x]?) for CUDA zero-copy."
                )
            }
        }
    }
}
