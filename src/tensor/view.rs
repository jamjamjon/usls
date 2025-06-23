//! Tensor view operations for zero-copy slicing and memory-efficient operations.
//!
//! This module provides `TensorView` and `TensorViewMut` for zero-copy tensor slicing,
//! which is crucial for performance optimization in KV cache operations and memory management.

use super::macros::{tensor_view_match, tensor_view_mut_match};
use super::slice::{IntoSliceSpec, SliceOrIndex};
use super::{DTypeTensor, Tensor, TensorElement};
use crate::core::DType;
use anyhow::Result;
use ndarray::{ArrayView, ArrayViewMut, Axis, IxDyn, SliceInfo, SliceInfoElem};
use std::ops::Range;

/// Zero-copy immutable view into a tensor
///
/// `TensorView` provides a lightweight, zero-copy view into tensor data without
/// allocating new memory. This is essential for performance-critical operations
/// like KV cache slicing and memory-efficient tensor operations.
///
/// # Examples
///
/// ```rust
/// use usls::tensor::Tensor;
///
/// let tensor = Tensor::zeros(vec![4, 4]);
/// let view = tensor.slice(&[0..2, 0..2]).unwrap();
/// assert_eq!(view.shape(), &[2, 2]);
/// ```
#[derive(Debug)]
pub struct TensorView<'a> {
    /// Reference to the underlying tensor data
    data: TensorViewData<'a>,
    /// Data type of the tensor
    dtype: DType,
    /// Shape of the view
    shape: Vec<usize>,
}

/// Zero-copy mutable view into a tensor
///
/// `TensorViewMut` provides a lightweight, zero-copy mutable view into tensor data
/// without allocating new memory. This enables in-place modifications for operations
/// like KV cache updates.
///
/// # Examples
///
/// ```rust
/// use usls::tensor::Tensor;
///
/// let mut tensor = Tensor::zeros(vec![4, 4]);
/// let mut view = tensor.slice_mut(&[0..2, 0..2]).unwrap();
/// // Modify the view in-place (specific operations depend on implementation)
/// ```
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    /// Mutable reference to the underlying tensor data
    data: TensorViewMutData<'a>,
    /// Data type of the tensor
    dtype: DType,
    /// Shape of the view
    shape: Vec<usize>,
}

/// Internal enum for immutable view data
#[allow(dead_code)]
#[derive(Debug)]
enum TensorViewData<'a> {
    F32(ArrayView<'a, f32, IxDyn>),
    F64(ArrayView<'a, f64, IxDyn>),
    F16(ArrayView<'a, half::f16, IxDyn>),
    Bf16(ArrayView<'a, half::bf16, IxDyn>),
    I8(ArrayView<'a, i8, IxDyn>),
    I16(ArrayView<'a, i16, IxDyn>),
    I32(ArrayView<'a, i32, IxDyn>),
    I64(ArrayView<'a, i64, IxDyn>),
    U8(ArrayView<'a, u8, IxDyn>),
    U16(ArrayView<'a, u16, IxDyn>),
    U32(ArrayView<'a, u32, IxDyn>),
    U64(ArrayView<'a, u64, IxDyn>),
    Bool(ArrayView<'a, bool, IxDyn>),
}

/// Internal enum for mutable view data
#[derive(Debug)]
enum TensorViewMutData<'a> {
    F32(ArrayViewMut<'a, f32, IxDyn>),
    F64(ArrayViewMut<'a, f64, IxDyn>),
    F16(ArrayViewMut<'a, half::f16, IxDyn>),
    Bf16(ArrayViewMut<'a, half::bf16, IxDyn>),
    I8(ArrayViewMut<'a, i8, IxDyn>),
    I16(ArrayViewMut<'a, i16, IxDyn>),
    I32(ArrayViewMut<'a, i32, IxDyn>),
    I64(ArrayViewMut<'a, i64, IxDyn>),
    U8(ArrayViewMut<'a, u8, IxDyn>),
    U16(ArrayViewMut<'a, u16, IxDyn>),
    U32(ArrayViewMut<'a, u32, IxDyn>),
    U64(ArrayViewMut<'a, u64, IxDyn>),
    Bool(ArrayViewMut<'a, bool, IxDyn>),
}

impl<'a> TensorView<'a> {
    /// Create a new tensor view from tensor data and slice ranges
    ///
    /// # Arguments
    ///
    /// * `data` - Reference to the underlying tensor data
    /// * `dtype` - Data type of the tensor
    /// * `ranges` - Slice ranges for each dimension
    ///
    /// # Returns
    ///
    /// Returns a `Result<TensorView>` containing the view or an error if slicing fails
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The number of ranges doesn't match tensor dimensions
    /// - Any range is out of bounds
    /// - The tensor data type is not supported
    pub fn new(data: &'a DTypeTensor, dtype: DType, ranges: &[Range<usize>]) -> Result<Self> {
        let view_data = match data {
            DTypeTensor::F32(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewData::F32(arr.slice(slice_info))
            }
            DTypeTensor::F64(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewData::F64(arr.slice(slice_info))
            }
            DTypeTensor::F16(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewData::F16(arr.slice(slice_info))
            }
            DTypeTensor::Bf16(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewData::Bf16(arr.slice(slice_info))
            }
            DTypeTensor::I8(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewData::I8(arr.slice(slice_info))
            }
            DTypeTensor::I32(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewData::I32(arr.slice(slice_info))
            }
            DTypeTensor::I64(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewData::I64(arr.slice(slice_info))
            }
            DTypeTensor::U8(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewData::U8(arr.slice(slice_info))
            }
            DTypeTensor::U32(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewData::U32(arr.slice(slice_info))
            }
            _ => anyhow::bail!("Tensor view not supported for dtype: {:?}", dtype),
        };

        let shape = get_view_shape(&view_data);

        Ok(Self {
            data: view_data,
            dtype,
            shape,
        })
    }

    /// Get the shape of the tensor view
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the data type of the tensor view
    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Create a new tensor view with slice specifications that support indexing
    ///
    /// # Arguments
    ///
    /// * `data` - Reference to the tensor data
    /// * `dtype` - Data type of the tensor
    /// * `slice_specs` - Slice specifications including ranges and indices
    ///
    /// # Returns
    ///
    /// Returns a `Result<TensorView>` containing the view or an error if creation fails
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The number of slice specs doesn't match tensor dimensions
    /// - Any range is out of bounds
    /// - The tensor data type is not supported
    pub fn new_with_specs(
        data: &'a DTypeTensor,
        dtype: DType,
        slice_specs: &[crate::tensor::slice::SliceOrIndex],
    ) -> Result<Self> {
        let view_data = match data {
            DTypeTensor::F32(arr) => {
                let slice_info = create_slice_info_with_indices(slice_specs, arr.shape())?;
                TensorViewData::F32(arr.slice(slice_info))
            }
            DTypeTensor::F64(arr) => {
                let slice_info = create_slice_info_with_indices(slice_specs, arr.shape())?;
                TensorViewData::F64(arr.slice(slice_info))
            }
            DTypeTensor::F16(arr) => {
                let slice_info = create_slice_info_with_indices(slice_specs, arr.shape())?;
                TensorViewData::F16(arr.slice(slice_info))
            }
            DTypeTensor::Bf16(arr) => {
                let slice_info = create_slice_info_with_indices(slice_specs, arr.shape())?;
                TensorViewData::Bf16(arr.slice(slice_info))
            }
            DTypeTensor::I8(arr) => {
                let slice_info = create_slice_info_with_indices(slice_specs, arr.shape())?;
                TensorViewData::I8(arr.slice(slice_info))
            }
            DTypeTensor::I32(arr) => {
                let slice_info = create_slice_info_with_indices(slice_specs, arr.shape())?;
                TensorViewData::I32(arr.slice(slice_info))
            }
            DTypeTensor::I64(arr) => {
                let slice_info = create_slice_info_with_indices(slice_specs, arr.shape())?;
                TensorViewData::I64(arr.slice(slice_info))
            }
            DTypeTensor::U8(arr) => {
                let slice_info = create_slice_info_with_indices(slice_specs, arr.shape())?;
                TensorViewData::U8(arr.slice(slice_info))
            }
            DTypeTensor::U32(arr) => {
                let slice_info = create_slice_info_with_indices(slice_specs, arr.shape())?;
                TensorViewData::U32(arr.slice(slice_info))
            }
            _ => anyhow::bail!("Tensor view not supported for dtype: {:?}", dtype),
        };

        let shape = get_view_shape(&view_data);

        Ok(Self {
            data: view_data,
            dtype,
            shape,
        })
    }

    /// Get the number of dimensions
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if the view is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.shape.iter().any(|&d| d == 0)
    }

    /// Reshape the tensor view to a new shape
    ///
    /// This method creates a new tensor with the specified shape. The total number
    /// of elements must remain the same.
    ///
    /// # Arguments
    ///
    /// * `shape` - The new shape as a tuple or slice
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` with the new shape or an error if incompatible
    pub fn to_shape<S>(&self, shape: S) -> Result<Tensor>
    where
        S: Into<Vec<usize>>,
    {
        let new_shape: Vec<usize> = shape.into();
        let new_len: usize = new_shape.iter().product();

        if new_len != self.len() {
            anyhow::bail!(
                "Cannot reshape tensor from {:?} to {:?}: incompatible sizes ({} vs {})",
                self.shape,
                new_shape,
                self.len(),
                new_len
            );
        }

        // Convert to owned tensor and reshape
        let owned_tensor = self.to_owned()?;
        owned_tensor.reshape(new_shape)
    }

    /// Convert the view to an owned tensor
    ///
    /// This creates a new tensor with copied data from the view.
    /// Use this when you need to own the data or when the view's lifetime ends.
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the owned tensor or an error
    pub fn to_owned(&self) -> Result<Tensor> {
        tensor_view_match!(&self.data, view, {
            let owned = view.to_owned().into_dyn();
            Ok(owned.into())
        })
    }

    // ========== Mathematical Operations ==========

    /// Element-wise addition with another tensor view
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor view to add
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn add(&self, other: &Self) -> Result<Tensor> {
        let self_tensor = self.to_owned()?;
        let other_tensor = other.to_owned()?;
        self_tensor + &other_tensor
    }

    /// Element-wise multiplication with another tensor view
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor view to multiply
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn mul(&self, other: &Self) -> Result<Tensor> {
        let self_tensor = self.to_owned()?;
        let other_tensor = other.to_owned()?;
        self_tensor * &other_tensor
    }

    /// Matrix multiplication with another tensor view
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor view to multiply (must be 2D)
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn matmul(&self, other: &Self) -> Result<Tensor> {
        let self_tensor = self.to_owned()?;
        let other_tensor = other.to_owned()?;
        self_tensor.matmul(&other_tensor)
    }

    /// Dot product with another tensor view
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor view for dot product
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn dot(&self, other: &Self) -> Result<Tensor> {
        let self_tensor = self.to_owned()?;
        let other_tensor = other.to_owned()?;
        self_tensor.dot(&other_tensor)
    }

    // ========== Aggregation Operations ==========

    /// Sum along specified dimension
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to sum along (None for all dimensions)
    /// * `keepdim` - Whether to keep the reduced dimension
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn sum(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.sum_dim(dim, keepdim)
    }

    /// Mean along specified dimension
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to compute mean along (None for all dimensions)
    /// * `keepdim` - Whether to keep the reduced dimension
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn mean(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.mean_dim(dim, keepdim)
    }

    /// Maximum along specified dimension
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to find max along (None for all dimensions)
    /// * `keepdim` - Whether to keep the reduced dimension
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn max(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.max_dim(dim, keepdim)
    }

    /// Minimum along specified dimension
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to find min along (None for all dimensions)
    /// * `keepdim` - Whether to keep the reduced dimension
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn min(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.min_dim(dim, keepdim)
    }

    /// Argmax along specified dimension
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to find argmax along (None for all dimensions)
    /// * `keepdim` - Whether to keep the reduced dimension
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn argmax(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.argmax(dim, keepdim)
    }

    /// Argmin along specified dimension
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to find argmin along (None for all dimensions)
    /// * `keepdim` - Whether to keep the reduced dimension
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn argmin(&self, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.argmin(dim, keepdim)
    }

    // ========== Unary Operations ==========

    /// Apply ReLU activation function
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn relu(&self) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.relu()
    }

    /// Apply exponential function
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn exp(&self) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.exp()
    }

    /// Apply natural logarithm
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn log(&self) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.log()
    }

    /// Apply absolute value
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn abs(&self) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.abs()
    }

    /// Apply negation
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn neg(&self) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.neg()
    }

    /// Apply sigmoid activation function
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn sigmoid(&self) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.sigmoid()
    }

    /// Apply SiLU activation function
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn silu(&self) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.silu()
    }

    /// Apply softmax along specified dimension
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to apply softmax along
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the result or an error
    pub fn softmax(&self, dim: usize) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.softmax(dim)
    }

    // ========== Shape Operations ==========

    /// Transpose the tensor view
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the transposed tensor or an error
    pub fn transpose(&self) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.transpose()
    }

    /// Reverse the order of axes (dimensions)
    ///
    /// This method reverses the order of tensor dimensions. For example,
    /// a tensor view with shape [2, 3, 4] becomes [4, 3, 2] after calling this method.
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the tensor with reversed axes or an error
    ///
    /// # Examples
    ///
    /// ```rust
    /// use usls::tensor::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![2, 3, 4]);
    /// let view = tensor.view();
    /// let reversed = view.reversed_axes().unwrap();
    /// assert_eq!(reversed.shape(), &[4, 3, 2]);
    /// ```
    pub fn reversed_axes(&self) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.reversed_axes()
    }

    /// Split the tensor view along the specified axis at the given index
    ///
    /// This method splits the tensor view into two parts along the specified axis.
    /// The first part contains elements before the split index, and the second part
    /// contains elements from the split index onwards.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to split (0-indexed)
    /// * `index` - The index at which to split (must be <= dimension size)
    ///
    /// # Returns
    ///
    /// Returns a `Result<(Tensor, Tensor)>` containing the two split tensors or an error
    ///
    /// # Panics
    ///
    /// Panics if axis or index is out of bounds.
    pub fn split_at(&self, axis: usize, index: usize) -> Result<(Tensor, Tensor)> {
        // Check if axis is valid
        if axis >= self.ndim() {
            anyhow::bail!(
                "Axis {} is out of bounds for tensor with {} dimensions",
                axis,
                self.ndim()
            );
        }

        // Check if index is valid
        let dim_size = self.shape()[axis];
        if index > dim_size {
            anyhow::bail!(
                "Index {} is out of bounds for axis {} with size {}",
                index,
                axis,
                dim_size
            );
        }

        // Use ndarray's efficient split_at method directly on views
        match &self.data {
            TensorViewData::F32(view) => {
                let (left_view, right_view) = view.clone().split_at(Axis(axis), index);
                let _left_shape = left_view.shape().to_vec();
                let _right_shape = right_view.shape().to_vec();
                Ok((
                    Tensor {
                        data: DTypeTensor::F32(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Tensor::generate_uid(),
                    },
                    Tensor {
                        data: DTypeTensor::F32(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Tensor::generate_uid(),
                    },
                ))
            }
            TensorViewData::F64(view) => {
                let (left_view, right_view) = view.clone().split_at(Axis(axis), index);
                let _left_shape = left_view.shape().to_vec();
                let _right_shape = right_view.shape().to_vec();
                Ok((
                    Tensor {
                        data: DTypeTensor::F64(left_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Tensor::generate_uid(),
                    },
                    Tensor {
                        data: DTypeTensor::F64(right_view.to_owned().into_shared()),
                        dtype: self.dtype,
                        uid: Tensor::generate_uid(),
                    },
                ))
            }
            // Add other data types as needed...
            _ => {
                // Fallback to the old implementation for now
                let tensor = self.to_owned()?;
                let (left, right) = tensor.split_at(axis, index)?;
                Ok((left, right))
            }
        }
    }

    /// Permute dimensions according to the given axes
    ///
    /// # Arguments
    ///
    /// * `axes` - New order of dimensions
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the permuted tensor or an error
    pub fn permute(&self, axes: &[usize]) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.permute(axes)
    }

    /// Reshape the tensor view to a new shape
    ///
    /// # Arguments
    ///
    /// * `shape` - New shape for the tensor
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the reshaped tensor or an error
    pub fn reshape<S: Into<Vec<usize>>>(&self, shape: S) -> Result<Tensor> {
        let new_shape = shape.into();

        // Check if total elements match
        let current_len = self.len();
        let new_len: usize = new_shape.iter().product();

        if current_len != new_len {
            anyhow::bail!(
                "Cannot reshape tensor from {:?} to {:?}: incompatible sizes ({} vs {})",
                self.shape,
                new_shape,
                current_len,
                new_len
            );
        }

        // Use the existing to_shape method
        self.to_shape(new_shape)
    }

    /// Remove dimensions of size 1
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the squeezed tensor or an error
    pub fn squeeze(&self) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.squeeze()
    }

    /// Remove a specific dimension of size 1
    ///
    /// This method removes a dimension at the specified axis if it has size 1.
    /// It's particularly useful for dimension iteration where we need to remove
    /// singleton dimensions created by slicing.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension index to remove
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` with the specified dimension removed
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The dimension index is out of bounds
    /// - The dimension size is not 1
    ///
    /// # Examples
    ///
    /// ```rust
    /// use usls::tensor::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![3, 1, 4]);
    /// let view = tensor.view();
    /// let squeezed = view.squeeze_dim(1).unwrap();
    /// assert_eq!(squeezed.shape(), &[3, 4]);
    /// ```
    pub fn squeeze_dim(&self, dim: usize) -> Result<Tensor> {
        if dim >= self.shape.len() {
            anyhow::bail!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                self.shape.len()
            );
        }

        if self.shape[dim] != 1 {
            anyhow::bail!("Cannot squeeze dimension {} with size {}, only dimensions of size 1 can be squeezed", dim, self.shape[dim]);
        }

        // Convert to tensor and squeeze the specified dimension
        let tensor = self.to_owned()?;
        tensor.squeeze_dim(dim)
    }

    /// Add a dimension of size 1 at the specified axis
    ///
    /// # Arguments
    ///
    /// * `axis` - Position to insert the new dimension
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the unsqueezed tensor or an error
    pub fn unsqueeze(&self, axis: usize) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        tensor.unsqueeze(axis)
    }

    // ========== Slicing Operations ==========

    /// Dynamic slicing with SliceOrIndex
    ///
    /// # Arguments
    ///
    /// * `slices` - Array of slice specifications
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the sliced tensor or an error
    pub fn slice_dyn(&self, slices: &[SliceOrIndex]) -> Result<Tensor> {
        let tensor = self.to_owned()?;
        // Use the slice() method and convert TensorView to Tensor
        let view = tensor.slice(slices)?;
        view.to_owned()
    }

    // ===== NDARRAY COMPATIBILITY METHODS =====
    // These methods provide compatibility with ndarray APIs used throughout the codebase

    /// Get raw vector data and offset (ndarray compatibility)
    ///
    /// Convert tensor view to a vector of specified type
    ///
    /// This method provides type-safe conversion from TensorView to Vec<T>
    /// where T implements TensorElement trait.
    pub fn to_vec<T: TensorElement>(&self) -> Result<Vec<T>> {
        match &self.data {
            TensorViewData::F32(view) => {
                // Always use safe conversion through T::from_f32
                let vec: Vec<T> = view.iter().map(|&x| T::from_f32(x)).collect();
                Ok(vec)
            }
            TensorViewData::F64(view) => {
                let vec: Vec<T> = view.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            TensorViewData::F16(view) => {
                let vec: Vec<T> = view.iter().map(|&x| T::from_f32(x.to_f32())).collect();
                Ok(vec)
            }
            TensorViewData::Bf16(view) => {
                let vec: Vec<T> = view.iter().map(|&x| T::from_f32(x.to_f32())).collect();
                Ok(vec)
            }
            TensorViewData::I8(view) => {
                let vec: Vec<T> = view.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            TensorViewData::I16(view) => {
                let vec: Vec<T> = view.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            TensorViewData::I32(view) => {
                let vec: Vec<T> = view.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            TensorViewData::I64(view) => {
                let vec: Vec<T> = view.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            TensorViewData::U8(view) => {
                let vec: Vec<T> = view.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            TensorViewData::U16(view) => {
                let vec: Vec<T> = view.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            TensorViewData::U32(view) => {
                let vec: Vec<T> = view.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            TensorViewData::U64(view) => {
                let vec: Vec<T> = view.iter().map(|&x| T::from_f32(x as f32)).collect();
                Ok(vec)
            }
            TensorViewData::Bool(view) => {
                let vec: Vec<T> = view
                    .iter()
                    .map(|&x| T::from_f32(if x { 1.0 } else { 0.0 }))
                    .collect();
                Ok(vec)
            }
        }
    }

    /// This method provides compatibility with `into_raw_vec_and_offset()` used in models
    ///
    /// **DEPRECATED**: Use `to_vec::<u8>()` instead for type-safe conversion
    #[deprecated(since = "0.1.0", note = "Use to_vec::<u8>() instead")]
    pub fn into_raw_vec_and_offset(self) -> Result<(Vec<u8>, usize)> {
        match self.data {
            TensorViewData::F32(view) => {
                let owned = view.to_owned();
                let (vec, offset) = owned.into_raw_vec_and_offset();
                let byte_vec = vec.into_iter().flat_map(|f| f.to_le_bytes()).collect();
                Ok((byte_vec, offset.unwrap_or(0) * 4)) // 4 bytes per f32
            }
            TensorViewData::F64(view) => {
                let owned = view.to_owned();
                let (vec, offset) = owned.into_raw_vec_and_offset();
                let byte_vec = vec.into_iter().flat_map(|f| f.to_le_bytes()).collect();
                Ok((byte_vec, offset.unwrap_or(0) * 8)) // 8 bytes per f64
            }
            TensorViewData::U8(view) => {
                let owned = view.to_owned();
                let (vec, offset) = owned.into_raw_vec_and_offset();
                Ok((vec, offset.unwrap_or(0)))
            }
            TensorViewData::I32(view) => {
                let owned = view.to_owned();
                let (vec, offset) = owned.into_raw_vec_and_offset();
                let byte_vec = vec.into_iter().flat_map(|i| i.to_le_bytes()).collect();
                Ok((byte_vec, offset.unwrap_or(0) * 4)) // 4 bytes per i32
            }
            _ => anyhow::bail!("into_raw_vec_and_offset not implemented for this data type yet"),
        }
    }

    /// Get dimensions as tuple (ndarray compatibility)
    ///
    /// Returns dimensions as a tuple for 1D, 2D, 3D tensors
    pub fn dim(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Get f32 data as slice (for compatibility with logits_sampler)
    ///
    /// This provides direct access to f32 data for performance-critical operations
    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        match &self.data {
            TensorViewData::F32(view) => view
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("Cannot get slice from non-contiguous array")),
            _ => anyhow::bail!("as_f32_slice only supports F32 tensors"),
        }
    }

    /// Apply element-wise mapping function (ndarray compatibility)
    ///
    /// This provides compatibility with `mapv()` method used in models
    pub fn mapv<F, B>(&self, f: F) -> Result<ndarray::Array<B, IxDyn>>
    where
        F: Fn(f32) -> B + Send + Sync,
        B: Clone + Send + Sync + 'static,
    {
        match &self.data {
            TensorViewData::F32(view) => Ok(view.mapv(f)),
            _ => anyhow::bail!("mapv currently only supports F32 tensors"),
        }
    }

    /// Create a slice view from this tensor view
    ///
    /// # Arguments
    ///
    /// * `spec` - Slice specification that can be converted to slice ranges
    ///
    /// # Returns
    ///
    /// Returns a `Result<TensorView>` containing the sliced view or an error
    pub fn slice<S: IntoSliceSpec>(&self, spec: S) -> Result<TensorView<'_>> {
        let slice_spec = spec.into_slice_spec();
        let slice_info = create_slice_info_with_indices(&slice_spec, &self.shape)?;

        let sliced_data = match &self.data {
            TensorViewData::F32(view) => TensorViewData::F32(view.slice(slice_info)),
            TensorViewData::F64(view) => TensorViewData::F64(view.slice(slice_info)),
            TensorViewData::F16(view) => TensorViewData::F16(view.slice(slice_info)),
            TensorViewData::Bf16(view) => TensorViewData::Bf16(view.slice(slice_info)),
            TensorViewData::I8(view) => TensorViewData::I8(view.slice(slice_info)),
            TensorViewData::I16(view) => TensorViewData::I16(view.slice(slice_info)),
            TensorViewData::I32(view) => TensorViewData::I32(view.slice(slice_info)),
            TensorViewData::I64(view) => TensorViewData::I64(view.slice(slice_info)),
            TensorViewData::U8(view) => TensorViewData::U8(view.slice(slice_info)),
            TensorViewData::U16(view) => TensorViewData::U16(view.slice(slice_info)),
            TensorViewData::U32(view) => TensorViewData::U32(view.slice(slice_info)),
            TensorViewData::U64(view) => TensorViewData::U64(view.slice(slice_info)),
            TensorViewData::Bool(view) => TensorViewData::Bool(view.slice(slice_info)),
        };

        let new_shape = get_view_shape(&sliced_data);

        Ok(TensorView {
            data: sliced_data,
            shape: new_shape,
            dtype: self.dtype,
        })
    }
}

impl<'a> TensorViewMut<'a> {
    /// Create a new mutable tensor view from tensor data and slice ranges
    ///
    /// # Arguments
    ///
    /// * `data` - Mutable reference to the underlying tensor data
    /// * `dtype` - Data type of the tensor
    /// * `ranges` - Slice ranges for each dimension
    ///
    /// # Returns
    ///
    /// Returns a `Result<TensorViewMut>` containing the mutable view or an error if slicing fails
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The number of ranges doesn't match tensor dimensions
    /// - Any range is out of bounds
    /// - The tensor data type is not supported
    pub fn new(data: &'a mut DTypeTensor, dtype: DType, ranges: &[Range<usize>]) -> Result<Self> {
        let view_data = match data {
            DTypeTensor::F32(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewMutData::F32(arr.slice_mut(slice_info))
            }
            DTypeTensor::F64(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewMutData::F64(arr.slice_mut(slice_info))
            }
            DTypeTensor::F16(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewMutData::F16(arr.slice_mut(slice_info))
            }
            DTypeTensor::Bf16(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewMutData::Bf16(arr.slice_mut(slice_info))
            }
            DTypeTensor::I8(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewMutData::I8(arr.slice_mut(slice_info))
            }
            DTypeTensor::I16(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewMutData::I16(arr.slice_mut(slice_info))
            }
            DTypeTensor::I32(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewMutData::I32(arr.slice_mut(slice_info))
            }
            DTypeTensor::I64(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewMutData::I64(arr.slice_mut(slice_info))
            }
            DTypeTensor::U8(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewMutData::U8(arr.slice_mut(slice_info))
            }
            DTypeTensor::U16(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewMutData::U16(arr.slice_mut(slice_info))
            }
            DTypeTensor::U32(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewMutData::U32(arr.slice_mut(slice_info))
            }
            DTypeTensor::U64(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewMutData::U64(arr.slice_mut(slice_info))
            }
            DTypeTensor::Bool(arr) => {
                let slice_info = create_slice_info(ranges, arr.shape())?;
                TensorViewMutData::Bool(arr.slice_mut(slice_info))
            }
        };

        let shape = get_view_mut_shape(&view_data);

        Ok(Self {
            data: view_data,
            dtype,
            shape,
        })
    }

    /// Create a new mutable tensor view from tensor data and slice specifications
    ///
    /// # Arguments
    ///
    /// * `data` - Mutable reference to the underlying tensor data
    /// * `dtype` - Data type of the tensor
    /// * `specs` - Slice specifications for each dimension
    ///
    /// # Returns
    ///
    /// Returns a `Result<TensorViewMut>` containing the mutable view or an error if slicing fails
    pub fn new_with_specs(
        data: &'a mut DTypeTensor,
        dtype: DType,
        specs: &[SliceOrIndex],
    ) -> Result<Self> {
        let view_data = match data {
            DTypeTensor::F32(arr) => {
                let slice_info = create_slice_info_with_indices(specs, arr.shape())?;
                TensorViewMutData::F32(arr.slice_mut(slice_info))
            }
            DTypeTensor::F64(arr) => {
                let slice_info = create_slice_info_with_indices(specs, arr.shape())?;
                TensorViewMutData::F64(arr.slice_mut(slice_info))
            }
            DTypeTensor::F16(arr) => {
                let slice_info = create_slice_info_with_indices(specs, arr.shape())?;
                TensorViewMutData::F16(arr.slice_mut(slice_info))
            }
            DTypeTensor::Bf16(arr) => {
                let slice_info = create_slice_info_with_indices(specs, arr.shape())?;
                TensorViewMutData::Bf16(arr.slice_mut(slice_info))
            }
            DTypeTensor::I8(arr) => {
                let slice_info = create_slice_info_with_indices(specs, arr.shape())?;
                TensorViewMutData::I8(arr.slice_mut(slice_info))
            }
            DTypeTensor::I16(arr) => {
                let slice_info = create_slice_info_with_indices(specs, arr.shape())?;
                TensorViewMutData::I16(arr.slice_mut(slice_info))
            }
            DTypeTensor::I32(arr) => {
                let slice_info = create_slice_info_with_indices(specs, arr.shape())?;
                TensorViewMutData::I32(arr.slice_mut(slice_info))
            }
            DTypeTensor::I64(arr) => {
                let slice_info = create_slice_info_with_indices(specs, arr.shape())?;
                TensorViewMutData::I64(arr.slice_mut(slice_info))
            }
            DTypeTensor::U8(arr) => {
                let slice_info = create_slice_info_with_indices(specs, arr.shape())?;
                TensorViewMutData::U8(arr.slice_mut(slice_info))
            }
            DTypeTensor::U16(arr) => {
                let slice_info = create_slice_info_with_indices(specs, arr.shape())?;
                TensorViewMutData::U16(arr.slice_mut(slice_info))
            }
            DTypeTensor::U32(arr) => {
                let slice_info = create_slice_info_with_indices(specs, arr.shape())?;
                TensorViewMutData::U32(arr.slice_mut(slice_info))
            }
            DTypeTensor::U64(arr) => {
                let slice_info = create_slice_info_with_indices(specs, arr.shape())?;
                TensorViewMutData::U64(arr.slice_mut(slice_info))
            }
            DTypeTensor::Bool(arr) => {
                let slice_info = create_slice_info_with_indices(specs, arr.shape())?;
                TensorViewMutData::Bool(arr.slice_mut(slice_info))
            }
        };

        let shape = get_view_mut_shape(&view_data);

        Ok(Self {
            data: view_data,
            dtype,
            shape,
        })
    }

    /// Get the shape of the tensor view
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the data type of the tensor view
    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the number of dimensions
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if the view is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.shape.iter().any(|&d| d == 0)
    }

    /// Fill the view with a scalar value
    ///
    /// This operation modifies the underlying tensor data in-place.
    ///
    /// # Arguments
    ///
    /// * `value` - The scalar value to fill with (will be cast to appropriate type)
    ///
    /// # Errors
    ///
    /// Returns an error if the data type doesn't support the fill operation
    pub fn fill(&mut self, value: f32) -> Result<()> {
        match &mut self.data {
            TensorViewMutData::F32(view) => {
                view.fill(value);
            }
            TensorViewMutData::F64(view) => {
                view.fill(value as f64);
            }
            TensorViewMutData::F16(view) => {
                view.fill(half::f16::from_f32(value));
            }
            TensorViewMutData::Bf16(view) => {
                view.fill(half::bf16::from_f32(value));
            }
            TensorViewMutData::I8(view) => {
                view.fill(value as i8);
            }
            TensorViewMutData::I16(view) => {
                view.fill(value as i16);
            }
            TensorViewMutData::I32(view) => {
                view.fill(value as i32);
            }
            TensorViewMutData::I64(view) => {
                view.fill(value as i64);
            }
            TensorViewMutData::U8(view) => {
                view.fill(value as u8);
            }
            TensorViewMutData::U16(view) => {
                view.fill(value as u16);
            }
            TensorViewMutData::U32(view) => {
                view.fill(value as u32);
            }
            TensorViewMutData::U64(view) => {
                view.fill(value as u64);
            }
            TensorViewMutData::Bool(view) => {
                view.fill(value != 0.0);
            }
        }
        Ok(())
    }

    /// Convert the mutable view to an owned tensor
    ///
    /// This creates a new tensor with copied data from the view.
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the owned tensor or an error
    pub fn to_owned(&self) -> Result<Tensor> {
        tensor_view_mut_match!(&self.data, view, {
            let owned = view.to_owned().into_dyn();
            Ok(owned.into())
        })
    }

    /// Fill with zeros
    ///
    /// # Returns
    ///
    /// Returns a `Result<()>` indicating success or failure
    pub fn zero_(&mut self) -> Result<()> {
        self.fill(0.0)
    }

    /// Fill with ones
    ///
    /// # Returns
    ///
    /// Returns a `Result<()>` indicating success or failure
    pub fn one_(&mut self) -> Result<()> {
        self.fill(1.0)
    }

    /// Reverse the order of axes in the tensor
    ///
    /// This method creates a new tensor with the axes in reverse order.
    /// For example, a tensor with shape [2, 3, 4] becomes [4, 3, 2].
    ///
    /// # Returns
    ///
    /// Returns a `Result<Tensor>` containing the tensor with reversed axes or an error
    ///
    pub fn reversed_axes(&self) -> Result<Tensor> {
        self.to_owned()?.reversed_axes()
    }

    /// Split the tensor view along the specified axis at the given index
    ///
    /// This method splits the tensor view into two parts along the specified axis.
    /// The first part contains elements before the split index, and the second part
    /// contains elements from the split index onwards.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to split (0-indexed)
    /// * `index` - The index at which to split (must be <= dimension size)
    ///
    /// # Returns
    ///
    /// Returns a `Result<(Tensor, Tensor)>` containing the two split tensors or an error
    ///
    /// # Panics
    ///
    /// Panics if axis or index is out of bounds.
    ///
    pub fn split_at(&self, axis: usize, index: usize) -> Result<(Tensor, Tensor)> {
        // Check if axis is valid
        if axis >= self.ndim() {
            anyhow::bail!(
                "Axis {} is out of bounds for tensor with {} dimensions",
                axis,
                self.ndim()
            );
        }

        // Check if index is valid
        let dim_size = self.shape()[axis];
        if index > dim_size {
            anyhow::bail!(
                "Index {} is out of bounds for axis {} with size {}",
                index,
                axis,
                dim_size
            );
        }

        // Use ndarray's efficient split_at method directly on views
        match &self.data {
            TensorViewMutData::F32(view) => {
                let (left_view, right_view) = view.view().split_at(Axis(axis), index);
                let left_shape = left_view.shape().to_vec();
                let right_shape = right_view.shape().to_vec();
                let left_tensor = TensorView {
                    data: TensorViewData::F32(left_view),
                    dtype: self.dtype,
                    shape: left_shape,
                }
                .to_owned()?;
                let right_tensor = TensorView {
                    data: TensorViewData::F32(right_view),
                    dtype: self.dtype,
                    shape: right_shape,
                }
                .to_owned()?;
                Ok((left_tensor, right_tensor))
            }
            TensorViewMutData::F64(view) => {
                let (left_view, right_view) = view.view().split_at(Axis(axis), index);
                let left_shape = left_view.shape().to_vec();
                let right_shape = right_view.shape().to_vec();
                let left_tensor = TensorView {
                    data: TensorViewData::F64(left_view),
                    dtype: self.dtype,
                    shape: left_shape,
                }
                .to_owned()?;
                let right_tensor = TensorView {
                    data: TensorViewData::F64(right_view),
                    dtype: self.dtype,
                    shape: right_shape,
                }
                .to_owned()?;
                Ok((left_tensor, right_tensor))
            }
            // Add other data types as needed...
            _ => {
                // Fallback to the old implementation for now
                let tensor = self.to_owned()?;
                tensor.split_at(axis, index)
            }
        }
    }

    /// Copy data from another tensor view
    ///
    /// # Arguments
    ///
    /// * `src` - Source tensor view to copy from
    ///
    /// # Returns
    ///
    /// Returns a `Result<()>` indicating success or failure
    pub fn copy_from(&mut self, src: &TensorView) -> Result<()> {
        if self.shape != src.shape {
            anyhow::bail!(
                "Shape mismatch: cannot copy from {:?} to {:?}",
                src.shape,
                self.shape
            );
        }

        match (&mut self.data, &src.data) {
            (TensorViewMutData::F32(dst), TensorViewData::F32(src)) => {
                dst.assign(src);
                Ok(())
            }
            (TensorViewMutData::F64(dst), TensorViewData::F64(src)) => {
                dst.assign(src);
                Ok(())
            }
            (TensorViewMutData::F16(dst), TensorViewData::F16(src)) => {
                dst.assign(src);
                Ok(())
            }
            (TensorViewMutData::Bf16(dst), TensorViewData::Bf16(src)) => {
                dst.assign(src);
                Ok(())
            }
            (TensorViewMutData::I8(dst), TensorViewData::I8(src)) => {
                dst.assign(src);
                Ok(())
            }
            (TensorViewMutData::I16(dst), TensorViewData::I16(src)) => {
                dst.assign(src);
                Ok(())
            }
            (TensorViewMutData::I32(dst), TensorViewData::I32(src)) => {
                dst.assign(src);
                Ok(())
            }
            (TensorViewMutData::I64(dst), TensorViewData::I64(src)) => {
                dst.assign(src);
                Ok(())
            }
            (TensorViewMutData::U8(dst), TensorViewData::U8(src)) => {
                dst.assign(src);
                Ok(())
            }
            (TensorViewMutData::U16(dst), TensorViewData::U16(src)) => {
                dst.assign(src);
                Ok(())
            }
            (TensorViewMutData::U32(dst), TensorViewData::U32(src)) => {
                dst.assign(src);
                Ok(())
            }
            (TensorViewMutData::U64(dst), TensorViewData::U64(src)) => {
                dst.assign(src);
                Ok(())
            }
            (TensorViewMutData::Bool(dst), TensorViewData::Bool(src)) => {
                dst.assign(src);
                Ok(())
            }
            _ => anyhow::bail!("Data type mismatch: cannot copy between different types"),
        }
    }

    /// Create a slice view from this mutable tensor view
    ///
    /// # Arguments
    ///
    /// * `spec` - Slice specification that can be converted to slice ranges
    ///
    /// # Returns
    ///
    /// Returns a `Result<TensorView>` containing the sliced view or an error
    pub fn slice<S: IntoSliceSpec>(&self, spec: S) -> Result<TensorView<'_>> {
        let slice_spec = spec.into_slice_spec();
        let shape = self.shape().to_vec();
        let slice_info = create_slice_info_with_indices(&slice_spec, &shape)?;

        let sliced_data = match &self.data {
            TensorViewMutData::F32(view) => TensorViewData::F32(view.slice(slice_info)),
            TensorViewMutData::F64(view) => TensorViewData::F64(view.slice(slice_info)),
            TensorViewMutData::F16(view) => TensorViewData::F16(view.slice(slice_info)),
            TensorViewMutData::Bf16(view) => TensorViewData::Bf16(view.slice(slice_info)),
            TensorViewMutData::I8(view) => TensorViewData::I8(view.slice(slice_info)),
            TensorViewMutData::I16(view) => TensorViewData::I16(view.slice(slice_info)),
            TensorViewMutData::I32(view) => TensorViewData::I32(view.slice(slice_info)),
            TensorViewMutData::I64(view) => TensorViewData::I64(view.slice(slice_info)),
            TensorViewMutData::U8(view) => TensorViewData::U8(view.slice(slice_info)),
            TensorViewMutData::U16(view) => TensorViewData::U16(view.slice(slice_info)),
            TensorViewMutData::U32(view) => TensorViewData::U32(view.slice(slice_info)),
            TensorViewMutData::U64(view) => TensorViewData::U64(view.slice(slice_info)),
            TensorViewMutData::Bool(view) => TensorViewData::Bool(view.slice(slice_info)),
        };

        let new_shape = get_view_shape(&sliced_data);

        Ok(TensorView {
            data: sliced_data,
            shape: new_shape,
            dtype: self.dtype,
        })
    }

    /// Create a mutable slice view from this mutable tensor view
    ///
    /// # Arguments
    ///
    /// * `spec` - Slice specification that can be converted to slice ranges
    ///
    /// # Returns
    ///
    /// Returns a `Result<TensorViewMut>` containing the sliced mutable view or an error
    pub fn slice_mut<S: IntoSliceSpec>(&mut self, spec: S) -> Result<TensorViewMut<'_>> {
        let slice_spec = spec.into_slice_spec();
        let shape = self.shape().to_vec();
        let slice_info = create_slice_info_with_indices(&slice_spec, &shape)?;

        let sliced_data = match &mut self.data {
            TensorViewMutData::F32(view) => TensorViewMutData::F32(view.slice_mut(slice_info)),
            TensorViewMutData::F64(view) => TensorViewMutData::F64(view.slice_mut(slice_info)),
            TensorViewMutData::F16(view) => TensorViewMutData::F16(view.slice_mut(slice_info)),
            TensorViewMutData::Bf16(view) => TensorViewMutData::Bf16(view.slice_mut(slice_info)),
            TensorViewMutData::I8(view) => TensorViewMutData::I8(view.slice_mut(slice_info)),
            TensorViewMutData::I16(view) => TensorViewMutData::I16(view.slice_mut(slice_info)),
            TensorViewMutData::I32(view) => TensorViewMutData::I32(view.slice_mut(slice_info)),
            TensorViewMutData::I64(view) => TensorViewMutData::I64(view.slice_mut(slice_info)),
            TensorViewMutData::U8(view) => TensorViewMutData::U8(view.slice_mut(slice_info)),
            TensorViewMutData::U16(view) => TensorViewMutData::U16(view.slice_mut(slice_info)),
            TensorViewMutData::U32(view) => TensorViewMutData::U32(view.slice_mut(slice_info)),
            TensorViewMutData::U64(view) => TensorViewMutData::U64(view.slice_mut(slice_info)),
            TensorViewMutData::Bool(view) => TensorViewMutData::Bool(view.slice_mut(slice_info)),
        };

        let new_shape = get_view_mut_shape(&sliced_data);

        Ok(TensorViewMut {
            data: sliced_data,
            shape: new_shape,
            dtype: self.dtype,
        })
    }
}

/// Helper function to create slice info from ranges
fn create_slice_info(
    ranges: &[Range<usize>],
    shape: &[usize],
) -> Result<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>> {
    if ranges.len() != shape.len() {
        anyhow::bail!(
            "Number of ranges ({}) doesn't match tensor dimensions ({})",
            ranges.len(),
            shape.len()
        );
    }

    let mut slice_elems = Vec::new();
    for (i, range) in ranges.iter().enumerate() {
        if range.end > shape[i] {
            anyhow::bail!(
                "Range end ({}) exceeds dimension size ({}) for axis {}",
                range.end,
                shape[i],
                i
            );
        }
        slice_elems.push(SliceInfoElem::Slice {
            start: range.start as isize,
            end: Some(range.end as isize),
            step: 1,
        });
    }

    SliceInfo::try_from(slice_elems)
        .map_err(|e| anyhow::anyhow!("Failed to create slice info: {}", e))
}

fn create_slice_info_with_indices(
    slice_specs: &[crate::tensor::slice::SliceOrIndex],
    shape: &[usize],
) -> Result<SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>> {
    use crate::tensor::slice::{resolve_negative_index, resolve_range};

    if slice_specs.len() > shape.len() {
        anyhow::bail!(
            "Number of slice specs ({}) exceeds tensor dimensions ({})",
            slice_specs.len(),
            shape.len()
        );
    }

    let mut slice_elems = Vec::new();

    // Process provided slice specs
    for (i, spec) in slice_specs.iter().enumerate() {
        match spec {
            crate::tensor::slice::SliceOrIndex::Range(range) => {
                let resolved_range = resolve_range(range.start, range.end, shape[i])?;
                slice_elems.push(SliceInfoElem::Slice {
                    start: resolved_range.start as isize,
                    end: Some(resolved_range.end as isize),
                    step: 1,
                });
            }
            crate::tensor::slice::SliceOrIndex::Index(idx) => {
                let resolved_idx = resolve_negative_index(*idx, shape[i])?;
                slice_elems.push(SliceInfoElem::Index(resolved_idx as isize));
            }
            crate::tensor::slice::SliceOrIndex::RangeFrom(start) => {
                let resolved_start = resolve_negative_index(*start, shape[i])?;
                slice_elems.push(SliceInfoElem::Slice {
                    start: resolved_start as isize,
                    end: Some(shape[i] as isize),
                    step: 1,
                });
            }
            crate::tensor::slice::SliceOrIndex::RangeTo(end) => {
                let resolved_end =
                    crate::tensor::slice::resolve_negative_index_for_range_end(*end, shape[i])?;
                slice_elems.push(SliceInfoElem::Slice {
                    start: 0,
                    end: Some(resolved_end as isize),
                    step: 1,
                });
            }
            crate::tensor::slice::SliceOrIndex::FullSlice => {
                slice_elems.push(SliceInfoElem::Slice {
                    start: 0,
                    end: Some(shape[i] as isize),
                    step: 1,
                });
            }
        }
    }

    // Fill remaining dimensions with full slices if needed
    for &dim_size in shape.iter().skip(slice_specs.len()) {
        slice_elems.push(SliceInfoElem::Slice {
            start: 0,
            end: Some(dim_size as isize),
            step: 1,
        });
    }

    Ok(unsafe { SliceInfo::new(slice_elems).unwrap() })
}

/// Helper function to get shape from immutable view data
fn get_view_shape(data: &TensorViewData) -> Vec<usize> {
    match data {
        TensorViewData::F32(view) => view.shape().to_vec(),
        TensorViewData::F64(view) => view.shape().to_vec(),
        TensorViewData::F16(view) => view.shape().to_vec(),
        TensorViewData::Bf16(view) => view.shape().to_vec(),
        TensorViewData::I8(view) => view.shape().to_vec(),
        TensorViewData::I16(view) => view.shape().to_vec(),
        TensorViewData::I32(view) => view.shape().to_vec(),
        TensorViewData::I64(view) => view.shape().to_vec(),
        TensorViewData::U8(view) => view.shape().to_vec(),
        TensorViewData::U16(view) => view.shape().to_vec(),
        TensorViewData::U32(view) => view.shape().to_vec(),
        TensorViewData::U64(view) => view.shape().to_vec(),
        TensorViewData::Bool(view) => view.shape().to_vec(),
    }
}

/// Helper function to get shape from mutable view data
fn get_view_mut_shape(data: &TensorViewMutData) -> Vec<usize> {
    match data {
        TensorViewMutData::F32(view) => view.shape().to_vec(),
        TensorViewMutData::F64(view) => view.shape().to_vec(),
        TensorViewMutData::F16(view) => view.shape().to_vec(),
        TensorViewMutData::Bf16(view) => view.shape().to_vec(),
        TensorViewMutData::I8(view) => view.shape().to_vec(),
        TensorViewMutData::I16(view) => view.shape().to_vec(),
        TensorViewMutData::I32(view) => view.shape().to_vec(),
        TensorViewMutData::I64(view) => view.shape().to_vec(),
        TensorViewMutData::U8(view) => view.shape().to_vec(),
        TensorViewMutData::U16(view) => view.shape().to_vec(),
        TensorViewMutData::U32(view) => view.shape().to_vec(),
        TensorViewMutData::U64(view) => view.shape().to_vec(),
        TensorViewMutData::Bool(view) => view.shape().to_vec(),
    }
}
