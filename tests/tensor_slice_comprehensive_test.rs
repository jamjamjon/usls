//!
//! Comprehensive tensor slicing test suite
//!
//! This module contains all tests for tensor slicing functionality,
//! including basic slicing, negative indexing, tuple syntax, edge cases, and performance tests.
//! Each test includes ndarray comparison to ensure 100% consistency with ndarray's s!() macro.

use anyhow::Result;
use ndarray::{s, Array};
use usls::tensor::Tensor;

// ============================================================================
// Basic Slicing Tests
// ============================================================================

#[test]
fn test_basic_slice_syntax() -> Result<()> {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![4, 6], data.clone())?;
    let ndarray = Array::from_shape_vec((4, 6), data)?;

    // Test basic range slicing
    let tensor_result = tensor.slice((1..3, 2..5))?;
    let ndarray_result = ndarray.slice(s![1..3, 2..5]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[2, 3]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    Ok(())
}

#[test]
fn test_full_slice_syntax() -> Result<()> {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![4, 6], data.clone())?;
    let ndarray = Array::from_shape_vec((4, 6), data)?;

    // Test full slice with ..
    let tensor_result = tensor.slice((.., ..))?;
    let ndarray_result = ndarray.slice(s![.., ..]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[4, 6]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    Ok(())
}

#[test]
fn test_range_from_syntax() -> Result<()> {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![4, 6], data.clone())?;
    let ndarray = Array::from_shape_vec((4, 6), data)?;

    // Test range from syntax
    let tensor_result = tensor.slice((2.., 3..))?;
    let ndarray_result = ndarray.slice(s![2.., 3..]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[2, 3]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    Ok(())
}

#[test]
fn test_range_to_syntax() -> Result<()> {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![4, 6], data.clone())?;
    let ndarray = Array::from_shape_vec((4, 6), data)?;

    // Test range to syntax
    let tensor_result = tensor.slice((..2, ..4))?;
    let ndarray_result = ndarray.slice(s![..2, ..4]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[2, 4]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    Ok(())
}

#[test]
fn test_index_selection() -> Result<()> {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![4, 6], data.clone())?;
    let ndarray = Array::from_shape_vec((4, 6), data)?;

    // Test index selection (reduces dimension)
    let tensor_result = tensor.slice((1, ..))?;
    let ndarray_result = ndarray.slice(s![1, ..]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[6]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    Ok(())
}

// ============================================================================
// Negative Indexing Tests
// ============================================================================

#[test]
fn test_negative_index_basic() -> Result<()> {
    let data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_shape_vec(vec![5], data.clone())?;
    let ndarray = Array::from_shape_vec(5, data)?;

    // Test -1 index (last element)
    let tensor_result = tensor.slice((-1,))?;
    let ndarray_result = ndarray.slice(s![-1]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    let empty_slice: &[usize] = &[];
    assert_eq!(tensor_result.shape(), empty_slice);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);
    assert_eq!(tensor_data[0], 4.0);

    Ok(())
}

#[test]
fn test_negative_range_from() -> Result<()> {
    let data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_shape_vec(vec![5], data.clone())?;
    let ndarray = Array::from_shape_vec(5, data)?;

    // Test -2.. (last two elements)
    let tensor_result = tensor.slice((-2..,))?;
    let ndarray_result = ndarray.slice(s![-2..]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[2]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);
    assert_eq!(tensor_data.as_slice(), &[3.0, 4.0]);

    Ok(())
}

#[test]
fn test_negative_range_to() -> Result<()> {
    let data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_shape_vec(vec![5], data.clone())?;
    let ndarray = Array::from_shape_vec(5, data)?;

    // Test ..-1 (all elements except last)
    let tensor_result = tensor.slice((..-1,))?;
    let ndarray_result = ndarray.slice(s![..-1]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[4]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);
    assert_eq!(tensor_data.as_slice(), &[0.0, 1.0, 2.0, 3.0]);

    Ok(())
}

#[test]
fn test_negative_range() -> Result<()> {
    let data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_shape_vec(vec![5], data.clone())?;
    let ndarray = Array::from_shape_vec(5, data)?;

    // Test -3..-1 (elements 2, 3)
    let tensor_result = tensor.slice((-3..-1,))?;
    let ndarray_result = ndarray.slice(s![-3..-1]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[2]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);
    assert_eq!(tensor_data.as_slice(), &[2.0, 3.0]);

    Ok(())
}

// ============================================================================
// 2D and 3D Negative Indexing Tests
// ============================================================================

#[test]
fn test_negative_index_2d() -> Result<()> {
    let data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let tensor = Tensor::from_shape_vec(vec![3, 3], data.clone())?;
    let ndarray = Array::from_shape_vec((3, 3), data)?;

    // Test last row: [-1, :]
    let tensor_result = tensor.slice((-1, ..))?;
    let ndarray_result = ndarray.slice(s![-1, ..]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[3]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);
    assert_eq!(tensor_data.as_slice(), &[6.0, 7.0, 8.0]);

    // Test last column: [:, -1]
    let tensor_result = tensor.slice((.., -1))?;
    let ndarray_result = ndarray.slice(s![.., -1]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[3]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);
    assert_eq!(tensor_data.as_slice(), &[2.0, 5.0, 8.0]);

    Ok(())
}

#[test]
fn test_negative_index_3d() -> Result<()> {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![2, 3, 4], data.clone())?;
    let ndarray = Array::from_shape_vec((2, 3, 4), data)?;

    // Test negative indexing in 3D: [-1, :, :]
    let tensor_result = tensor.slice((-1, .., ..))?;
    let ndarray_result = ndarray.slice(s![-1, .., ..]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[3, 4]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    // Test mixed negative indexing: [:, -1, :]
    let tensor_result = tensor.slice((.., -1, ..))?;
    let ndarray_result = ndarray.slice(s![.., -1, ..]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[2, 4]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    Ok(())
}

// ============================================================================
// Complex Tuple Syntax Tests
// ============================================================================

#[test]
fn test_tuple_mixed_positive_negative() -> Result<()> {
    let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![4, 5, 6], data.clone())?;
    let ndarray = Array::from_shape_vec((4, 5, 6), data)?;

    // Test mixed positive and negative ranges
    let tensor_result = tensor.slice((1..3, -2.., 0..3))?;
    let ndarray_result = ndarray.slice(s![1..3, -2.., 0..3]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[2, 2, 3]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    Ok(())
}

#[test]
fn test_tuple_negative_range_combinations() -> Result<()> {
    let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![3, 4, 5], data.clone())?;
    let ndarray = Array::from_shape_vec((3, 4, 5), data)?;

    // Test various negative range combinations
    let tensor_result = tensor.slice((-2.., -3.., -2..))?;
    let ndarray_result = ndarray.slice(s![-2.., -3.., -2..]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[2, 3, 2]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    Ok(())
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_empty_slice() -> Result<()> {
    let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![4, 5], data.clone())?;
    let ndarray = Array::from_shape_vec((4, 5), data)?;

    // Test empty slice
    let tensor_result = tensor.slice((0..0, ..))?;
    let ndarray_result = ndarray.slice(s![0..0, ..]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[0, 5]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    Ok(())
}

#[test]
fn test_single_element_slice() -> Result<()> {
    let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![4, 5], data.clone())?;
    let ndarray = Array::from_shape_vec((4, 5), data)?;

    // Test single element slice
    let tensor_result = tensor.slice((1..2, 2..3))?;
    let ndarray_result = ndarray.slice(s![1..2, 2..3]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[1, 1]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    Ok(())
}

#[test]
fn test_boundary_slicing() -> Result<()> {
    let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![5, 6], data.clone())?;
    let ndarray = Array::from_shape_vec((5, 6), data)?;

    // Test boundary slicing (valid indices within bounds)
    let tensor_result = tensor.slice((3..4, 4..5))?;
    let ndarray_result = ndarray.slice(s![3..4, 4..5]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[1, 1]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    Ok(())
}

// ============================================================================
// Chained Slicing Tests
// ============================================================================

#[test]
fn test_chained_slicing() -> Result<()> {
    let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![3, 4, 5], data.clone())?;
    let ndarray = Array::from_shape_vec((3, 4, 5), data)?;

    // Test chained slicing
    let step1_tensor = tensor.slice((0..2, .., ..))?;
    let step1_ndarray = ndarray.slice(s![0..2, .., ..]);

    let step2_tensor = step1_tensor.slice((.., 1..3, ..))?;
    let step2_ndarray = step1_ndarray.slice(s![.., 1..3, ..]);

    let final_tensor = step2_tensor.slice((.., .., 2..4))?;
    let final_ndarray = step2_ndarray.slice(s![.., .., 2..4]);

    assert_eq!(final_tensor.shape(), final_ndarray.shape());
    assert_eq!(final_tensor.shape(), &[2, 2, 2]);

    let tensor_data = final_tensor.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = final_ndarray.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    Ok(())
}

// ============================================================================
// Performance and Stress Tests
// ============================================================================

#[test]
fn test_large_tensor_slicing() -> Result<()> {
    let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![10, 10, 10], data.clone())?;
    let ndarray = Array::from_shape_vec((10, 10, 10), data)?;

    // Test slicing on larger tensor
    let tensor_result = tensor.slice((2..8, 3..7, 1..9))?;
    let ndarray_result = ndarray.slice(s![2..8, 3..7, 1..9]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[6, 4, 8]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    Ok(())
}

#[test]
fn test_complex_negative_patterns() -> Result<()> {
    let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![4, 5, 6], data.clone())?;
    let ndarray = Array::from_shape_vec((4, 5, 6), data)?;

    // Test complex negative indexing patterns
    let tensor_result = tensor.slice((-3..-1, -4.., ..-2))?;
    let ndarray_result = ndarray.slice(s![-3..-1, -4.., ..-2]);

    assert_eq!(tensor_result.shape(), ndarray_result.shape());
    assert_eq!(tensor_result.shape(), &[2, 4, 4]);

    let tensor_data = tensor_result.to_owned()?.to_vec::<f32>()?;
    let ndarray_binding = ndarray_result.to_owned();
    let ndarray_data = ndarray_binding.as_slice().unwrap();
    assert_eq!(tensor_data.as_slice(), ndarray_data);

    Ok(())
}
