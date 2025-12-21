use anyhow::Result;
use std::convert::TryFrom;

use crate::XAny;

// Implementation for ort_inputs! macro compatibility
impl<'a> TryFrom<&'a XAny> for ort::session::SessionInputValue<'a> {
    type Error = anyhow::Error;

    fn try_from(tensor: &'a XAny) -> Result<Self> {
        match tensor {
            XAny::Host(x) => {
                // Try zero-copy for f32 standard layout
                if x.0.is_standard_layout() {
                    let tensor_ref = ort::value::TensorRef::from_array_view(x.0.view())
                        .map_err(|e| anyhow::anyhow!("Failed to create TensorRef: {:?}", e))?;
                    Ok(ort::session::SessionInputValue::from(tensor_ref))
                } else {
                    // Fallback: create owned tensor
                    let value = ort::value::Value::from_array(x.0.clone())
                        .map_err(|e| anyhow::anyhow!("Failed to create Value: {:?}", e))?;
                    Ok(ort::session::SessionInputValue::from(value))
                }
            }
            #[cfg(feature = "cuda")]
            XAny::Device(cuda_tensor) => {
                // Auto-convert CUDA tensor to host for ort_inputs! compatibility
                let host_tensor = cuda_tensor.to_host()?;
                let value = ort::value::Value::from_array(host_tensor.0).map_err(|e| {
                    anyhow::anyhow!("Failed to create Value from CUDA tensor: {:?}", e)
                })?;
                Ok(ort::session::SessionInputValue::from(value))
            }
        }
    }
}

// For owned XAny (less common but needed for some patterns)
impl TryFrom<XAny> for ort::session::SessionInputValue<'static> {
    type Error = anyhow::Error;

    fn try_from(tensor: XAny) -> Result<Self> {
        match tensor {
            XAny::Host(x) => {
                let value = ort::value::Value::from_array(x.0)
                    .map_err(|e| anyhow::anyhow!("Failed to create Value: {:?}", e))?;
                Ok(ort::session::SessionInputValue::from(value))
            }
            #[cfg(feature = "cuda")]
            XAny::Device(cuda_tensor) => {
                // Auto-convert CUDA tensor to host for ort_inputs! compatibility
                let host_tensor = cuda_tensor.to_host()?;
                let value = ort::value::Value::from_array(host_tensor.0).map_err(|e| {
                    anyhow::anyhow!("Failed to create Value from CUDA tensor: {:?}", e)
                })?;
                Ok(ort::session::SessionInputValue::from(value))
            }
        }
    }
}
