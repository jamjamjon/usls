//! Shape manipulation operations for tensors
//!
//! This module provides comprehensive shape manipulation operations including
//! stack, reshape, flatten, squeeze, unsqueeze, and broadcasting operations.

use super::{DTypeTensor, Tensor};
use anyhow::Result;
use ndarray::IxDyn;

impl Tensor {
    /// Stack tensors along a new dimension
    ///
    /// Creates a new tensor by stacking the input tensors along a new dimension.
    /// All input tensors must have the same shape.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensors to stack
    /// * `dim` - Dimension along which to stack (0 <= dim <= ndim)
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` with shape where the new dimension is inserted at `dim`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use usls::tensor::Tensor;
    /// use usls::core::DType;
    ///
    /// let t1 = Tensor::ones_with_dtype(vec![2, 3], DType::Fp32);
    /// let t2 = Tensor::zeros_with_dtype(vec![2, 3], DType::Fp32);
    /// let stacked = Tensor::stack(&[t1, t2], 0).unwrap();
    /// assert_eq!(stacked.shape(), &[2, 2, 3]);
    /// ```
    pub fn stack(tensors: &[Self], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            anyhow::bail!("Cannot stack empty tensor array");
        }

        let first = &tensors[0];
        let shape = first.shape();
        let dtype = first.dtype();

        // Validate all tensors have same shape and dtype
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape() != shape {
                anyhow::bail!(
                    "All tensors must have the same shape for stacking. \
                     Tensor 0 has shape {:?}, but tensor {} has shape {:?}",
                    shape,
                    i,
                    tensor.shape()
                );
            }
            if tensor.dtype() != dtype {
                anyhow::bail!(
                    "All tensors must have the same dtype for stacking. \
                     Tensor 0 has dtype {:?}, but tensor {} has dtype {:?}",
                    dtype,
                    i,
                    tensor.dtype()
                );
            }
        }

        if dim > shape.len() {
            anyhow::bail!(
                "Stack dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            );
        }

        // Create new shape with inserted dimension
        let mut new_shape = shape.to_vec();
        new_shape.insert(dim, tensors.len());

        match &first.data {
            DTypeTensor::F32(_) => {
                let arrays: Result<Vec<_>> = tensors
                    .iter()
                    .map(|t| match &t.data {
                        DTypeTensor::F32(arr) => Ok(arr.clone()),
                        _ => anyhow::bail!("Inconsistent tensor data types"),
                    })
                    .collect();
                let arrays = arrays?;

                // Stack arrays along the specified dimension
                let stacked = stack_arrays_f32(&arrays, dim)?;
                Ok(stacked.into_dyn().into())
            }
            DTypeTensor::F64(_) => {
                let arrays: Result<Vec<_>> = tensors
                    .iter()
                    .map(|t| match &t.data {
                        DTypeTensor::F64(arr) => Ok(arr.clone()),
                        _ => anyhow::bail!("Inconsistent tensor data types"),
                    })
                    .collect();
                let arrays = arrays?;
                let stacked = stack_arrays_f64(&arrays, dim)?;
                Ok(stacked.into_dyn().into())
            }
            _ => anyhow::bail!("Stack operation not yet implemented for dtype: {:?}", dtype),
        }
    }

    /// Vertical stack (stack along dimension 0)
    ///
    /// Convenience method for stacking tensors along the first dimension.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensors to stack vertically
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` with the tensors stacked along dimension 0
    pub fn vstack(tensors: &[Self]) -> Result<Self> {
        Self::stack(tensors, 0)
    }

    /// Horizontal stack (stack along dimension 1)
    ///
    /// Convenience method for stacking tensors along the second dimension.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of tensors to stack horizontally
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` with the tensors stacked along dimension 1
    pub fn hstack(tensors: &[Self]) -> Result<Self> {
        Self::stack(tensors, 1)
    }

    /// Flatten tensor to 1D
    ///
    /// Flattens the tensor between the specified start and end dimensions.
    ///
    /// # Arguments
    ///
    /// * `start_dim` - First dimension to flatten (inclusive)
    /// * `end_dim` - Last dimension to flatten (inclusive)
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` with flattened dimensions
    pub fn flatten(&self, start_dim: usize, end_dim: usize) -> Result<Self> {
        let shape = self.shape();
        let ndim = shape.len();

        if start_dim >= ndim || end_dim >= ndim || start_dim > end_dim {
            anyhow::bail!(
                "Invalid flatten dimensions: start_dim={}, end_dim={}, ndim={}",
                start_dim,
                end_dim,
                ndim
            );
        }

        if start_dim == end_dim {
            // No flattening needed
            return Ok(self.clone());
        }

        // Calculate new shape
        let mut new_shape = Vec::new();

        // Add dimensions before start_dim
        new_shape.extend_from_slice(&shape[..start_dim]);

        // Calculate flattened dimension size
        let flattened_size: usize = shape[start_dim..=end_dim].iter().product();
        new_shape.push(flattened_size);

        // Add dimensions after end_dim
        if end_dim + 1 < ndim {
            new_shape.extend_from_slice(&shape[end_dim + 1..]);
        }

        self.clone().reshape(new_shape)
    }
}

// Helper functions for stacking arrays

fn stack_arrays_f32(
    arrays: &[ndarray::ArcArray<f32, IxDyn>],
    dim: usize,
) -> Result<ndarray::Array<f32, IxDyn>> {
    if arrays.is_empty() {
        return Err(anyhow::anyhow!("Cannot stack empty array list"));
    }

    // Use ndarray's stack function directly
    let views: Vec<_> = arrays.iter().map(|arr| arr.view()).collect();
    let stacked = ndarray::stack(ndarray::Axis(dim), &views)
        .map_err(|e| anyhow::anyhow!("Failed to stack arrays: {}", e))?;

    Ok(stacked)
}

fn stack_arrays_f64(
    arrays: &[ndarray::ArcArray<f64, IxDyn>],
    dim: usize,
) -> Result<ndarray::Array<f64, IxDyn>> {
    if arrays.is_empty() {
        return Err(anyhow::anyhow!("Cannot stack empty array list"));
    }

    // Use ndarray's stack function directly
    let views: Vec<_> = arrays.iter().map(|arr| arr.view()).collect();
    let stacked = ndarray::stack(ndarray::Axis(dim), &views)
        .map_err(|e| anyhow::anyhow!("Failed to stack arrays: {}", e))?;

    Ok(stacked)
}
