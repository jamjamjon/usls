//! Tensor dimension iteration utilities
//!
//! This module provides efficient iteration over tensor dimensions,
//! yielding zero-copy views for high-performance operations.

use crate::tensor::Tensor;
use rayon::prelude::*;

/// Mutable iterator over slices along a specific tensor dimension
///
/// This iterator provides efficient mutable access to tensor slices along a specified dimension,
/// yielding mutable `Tensor` instances. Similar to `TensorDimIter` but allows modification.
///
/// # Performance
///
/// - Efficient slicing: Uses tensor views internally where possible
/// - Lazy evaluation: Slices are created on-demand during iteration
/// - Memory efficient: Only one slice exists at a time
/// - Mutable access: Allows in-place modifications of tensor slices
///
/// # Examples
///
/// ```rust
/// use usls::tensor::Tensor;
///
/// let mut tensor = Tensor::zeros(vec![3, 4, 5]);
/// let mut iter = tensor.iter_mut_dim(0);
///
/// // Each iteration yields a mutable Tensor of shape [4, 5]
/// while let Some(mut slice_tensor) = iter.next_slice() {
///     assert_eq!(slice_tensor.shape(), &[4, 5]);
///     // Can modify the slice here
/// }
/// ```
pub struct TensorDimIterMut<'a> {
    tensor: &'a mut Tensor,
    dim: usize,
    current: usize,
    total: usize,
}

impl<'a> TensorDimIterMut<'a> {
    /// Create a new mutable dimension iterator
    ///
    /// # Arguments
    ///
    /// * `tensor` - The mutable tensor to iterate over
    /// * `dim` - The dimension index to iterate along
    ///
    /// # Panics
    ///
    /// Panics if `dim` is out of bounds for the tensor
    pub fn new(tensor: &'a mut Tensor, dim: usize) -> Self {
        assert!(dim < tensor.ndim(), "Dimension {} is out of bounds", dim);
        let total = tensor.shape()[dim];

        Self {
            tensor,
            dim,
            current: 0,
            total,
        }
    }

    /// Get the total number of slices that will be yielded
    pub fn total_len(&self) -> usize {
        self.total
    }

    /// Check if the iterator is empty
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Get the remaining number of slices
    pub fn remaining(&self) -> usize {
        self.total - self.current
    }
}

// Note: Iterator implementation for TensorDimIterMut is more complex due to
// Rust's borrowing rules. We'll implement it as a manual iteration method
// to avoid lifetime issues with mutable references.
impl<'a> TensorDimIterMut<'a> {
    /// Get the next mutable slice, if available
    ///
    /// This method manually advances the iterator and returns the next slice.
    /// Due to Rust's borrowing rules, we can't implement the standard Iterator trait
    /// for mutable references in this context.
    pub fn next_slice(&mut self) -> Option<crate::tensor::view::TensorViewMut<'_>> {
        if self.current >= self.total {
            return None;
        }

        // Create slice specs for slicing with index on the target dimension
        let mut slice_specs = Vec::with_capacity(self.tensor.ndim());

        for (i, &_size) in self.tensor.shape().iter().enumerate() {
            if i == self.dim {
                // For the target dimension, use index to remove the dimension
                slice_specs.push(crate::tensor::slice::SliceOrIndex::Index(
                    self.current as isize,
                ));
            } else {
                // For other dimensions, include the full range
                slice_specs.push(crate::tensor::slice::SliceOrIndex::FullSlice);
            }
        }

        // Create the mutable view with the squeezed dimension
        let view = self
            .tensor
            .slice_mut(slice_specs)
            .expect("Slice should be valid for dimension iteration");

        self.current += 1;
        Some(view)
    }

    /// Reset the iterator to the beginning
    pub fn reset(&mut self) {
        self.current = 0;
    }
}

/// Iterator over slices along a specific tensor dimension
///
/// This iterator provides efficient access to tensor slices along a specified dimension,
/// yielding `Tensor` instances. While not completely zero-copy due to current limitations,
/// it provides a clean API for dimension-wise iteration.
///
/// # Performance
///
/// - Efficient slicing: Uses tensor views internally where possible
/// - Lazy evaluation: Slices are created on-demand during iteration
/// - Memory efficient: Only one slice exists at a time
///
/// # Examples
///
/// ```rust
/// use usls::tensor::Tensor;
///
/// let tensor = Tensor::zeros(vec![3, 4, 5]);
/// let mut iter = tensor.iter_dim(0);
///
/// // Each iteration yields a Tensor of shape [4, 5]
/// while let Some(slice_tensor) = iter.next() {
///     assert_eq!(slice_tensor.shape(), &[4, 5]);
/// }
/// ```
pub struct TensorDimIter<'a> {
    tensor: &'a Tensor,
    dim: usize,
    current: usize,
    total: usize,
}

impl<'a> TensorDimIter<'a> {
    /// Create a new dimension iterator
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to iterate over
    /// * `dim` - The dimension index to iterate along
    ///
    /// # Panics
    ///
    /// Panics if `dim` is out of bounds for the tensor
    pub fn new(tensor: &'a Tensor, dim: usize) -> Self {
        assert!(dim < tensor.ndim(), "Dimension {} is out of bounds", dim);
        let total = tensor.shape()[dim];

        Self {
            tensor,
            dim,
            current: 0,
            total,
        }
    }

    /// Get the total number of slices that will be yielded
    pub fn total_len(&self) -> usize {
        self.total
    }

    /// Check if the iterator is empty
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Get the remaining number of slices
    pub fn remaining(&self) -> usize {
        self.total - self.current
    }
}

impl<'a> Iterator for TensorDimIter<'a> {
    type Item = Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.total {
            return None;
        }

        // Check if any dimension (except the iteration dimension) has size 0
        // If so, we need to handle empty slices specially
        let has_empty_dim = self
            .tensor
            .shape()
            .iter()
            .enumerate()
            .any(|(i, &size)| i != self.dim && size == 0);

        if has_empty_dim {
            // For empty dimensions, create an empty tensor with the correct shape
            let mut target_shape = self.tensor.shape().to_vec();
            target_shape.remove(self.dim);

            self.current += 1;
            return Some(Tensor::zeros(target_shape));
        }

        // Create ranges for slicing
        let mut ranges = Vec::with_capacity(self.tensor.ndim());

        for (i, &size) in self.tensor.shape().iter().enumerate() {
            if i == self.dim {
                // For the target dimension, slice only the current index
                ranges.push(self.current..self.current + 1);
            } else {
                // For other dimensions, include the full range
                ranges.push(0..size);
            }
        }

        // Create the view and remove the singleton dimension
        let view = self
            .tensor
            .slice(ranges)
            .expect("Slice should be valid for dimension iteration");

        // Remove the singleton dimension to get the expected shape
        let squeezed_tensor = view
            .squeeze_dim(self.dim)
            .expect("Squeeze should succeed for singleton dimension");

        self.current += 1;
        Some(squeezed_tensor)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for TensorDimIter<'a> {
    fn len(&self) -> usize {
        self.remaining()
    }
}

impl<'a> std::iter::FusedIterator for TensorDimIter<'a> {}

/// Parallel iterator implementation for TensorDimIter
///
/// This allows using `.par_iter()` and other parallel operations on TensorDimIter.
/// The parallel iteration is implemented by collecting all tensor slices first,
/// then creating a parallel iterator over the collected items.
impl IntoParallelIterator for TensorDimIter<'_> {
    type Item = Tensor;
    type Iter = rayon::vec::IntoIter<Tensor>;

    fn into_par_iter(self) -> Self::Iter {
        // Collect all tensor slices
        let tensors: Vec<Tensor> = self.collect();
        tensors.into_par_iter()
    }
}

/// Reference parallel iterator implementation for TensorDimIter
///
/// Note: This implementation consumes the iterator by collecting all remaining items.
/// For better performance with large tensors, consider using the owned version.
impl<'a> IntoParallelIterator for &'a mut TensorDimIter<'_> {
    type Item = Tensor;
    type Iter = rayon::vec::IntoIter<Tensor>;

    fn into_par_iter(self) -> Self::Iter {
        // Collect all remaining tensor slices
        let mut tensors = Vec::with_capacity(self.remaining());
        for tensor in self.by_ref() {
            tensors.push(tensor);
        }
        tensors.into_par_iter()
    }
}
