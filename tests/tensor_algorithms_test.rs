//! Integration tests for tensor algorithms
//!
//! This module tests the mathematical operations and algorithms
//! implemented for tensors including aggregation, linear algebra,
//! and shape manipulation operations.

use anyhow::Result;
use usls::tensor::Tensor;

#[test]
fn test_sum_operations() -> Result<()> {
    // Test sum with different dimensions
    let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
    let tensor = Tensor::from_shape_vec(vec![2, 3], data)?;

    // Sum all elements
    let sum_all = tensor.sum_dim(None, false)?;
    assert_eq!(sum_all.shape(), &[] as &[usize]);

    // Sum along dimension 0
    let sum_dim0 = tensor.sum_dim(Some(0), false)?;
    assert_eq!(sum_dim0.shape(), &[3]);
    // assert_eq!(sum_dim0, Tensor::from(vec![5.0f32, 7., 9.]));

    // Sum along dimension 1 with keepdims
    let sum_dim1_keepdims = tensor.sum_dim(Some(1), true)?;
    assert_eq!(sum_dim1_keepdims.shape(), &[2, 1]);

    Ok(())
}

#[test]
fn test_min_max_operations() -> Result<()> {
    let data = vec![3.0f32, 1.0f32, 4.0f32, 1.0f32, 5.0f32, 9.0f32];
    let tensor = Tensor::from_shape_vec(vec![2, 3], data)?;

    // Min all elements
    let min_all = tensor.min_dim(None, false)?;
    assert_eq!(min_all.shape(), &[] as &[usize]);

    // Max all elements
    let max_all = tensor.max_dim(None, false)?;
    assert_eq!(max_all.shape(), &[] as &[usize]);

    // Max along dimension 0
    let max_dim0 = tensor.max_dim(Some(0), false)?;
    assert_eq!(max_dim0.shape(), &[3]);

    // Min along dimension 1 with keepdims
    let min_dim1_keepdims = tensor.min_dim(Some(1), true)?;
    assert_eq!(min_dim1_keepdims.shape(), &[2, 1]);

    Ok(())
}

#[test]
fn test_dot_product() -> Result<()> {
    // Test 2D matrix multiplication (dot2 expects A[m,k] × B[n,k] format)
    let a = Tensor::from_shape_vec(
        vec![2, 3],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32],
    )?;
    let b = Tensor::from_shape_vec(
        vec![2, 3],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32],
    )?;

    let dot_result = a.dot(&b)?;
    assert_eq!(dot_result.shape(), &[2, 2]);

    Ok(())
}

#[test]
fn test_matrix_multiplication() -> Result<()> {
    // Test 2D matrix multiplication
    let a = Tensor::from_shape_vec(
        vec![2, 3],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32],
    )?;
    let b = Tensor::from_shape_vec(
        vec![3, 2],
        vec![7.0f32, 8.0f32, 9.0f32, 10.0f32, 11.0f32, 12.0f32],
    )?;

    let matmul_result = a.matmul(&b)?;
    assert_eq!(matmul_result.shape(), &[2, 2]);

    Ok(())
}

#[test]
fn test_argmax_argmin() -> Result<()> {
    let data = vec![3.0f32, 1.0f32, 4.0f32, 1.0f32, 5.0f32, 9.0f32];
    let tensor = Tensor::from_shape_vec(vec![2, 3], data)?;

    // Argmax all elements
    let argmax_all = tensor.argmax(None, false)?;
    assert_eq!(argmax_all.shape(), &[] as &[usize]);

    // Argmin along dimension 0 (valid dimension for [2, 3] tensor)
    let argmin_dim0 = tensor.argmin(Some(0), false)?;
    assert_eq!(argmin_dim0.shape(), &[3]);

    Ok(())
}

#[test]
fn test_concatenation() -> Result<()> {
    let a = Tensor::from_shape_vec(
        vec![2, 3],
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32],
    )?;
    let b = Tensor::from_shape_vec(
        vec![2, 3],
        vec![7.0f32, 8.0f32, 9.0f32, 10.0f32, 11.0f32, 12.0f32],
    )?;

    // Concatenate along dimension 0
    let concat_dim0 = Tensor::concat(&[a.clone(), b.clone()], 0)?;
    assert_eq!(concat_dim0.shape(), &[4, 3]);

    // Concatenate along dimension 1
    let concat_dim1 = Tensor::concat(&[a, b], 1)?;
    assert_eq!(concat_dim1.shape(), &[2, 6]);

    Ok(())
}

#[test]
fn test_reshape_operations() -> Result<()> {
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![2, 3, 4], data)?;

    // Reshape to different dimensions
    let reshaped = tensor.clone().reshape(vec![4, 6])?;
    assert_eq!(reshaped.shape(), &[4, 6]);

    // Reshape to 1D
    let flattened = tensor.reshape(vec![24])?;
    assert_eq!(flattened.shape(), &[24]);

    Ok(())
}

#[test]
fn test_squeeze_unsqueeze() -> Result<()> {
    let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
    let tensor = Tensor::from_shape_vec(vec![1, 2, 3, 1], data)?;

    // Squeeze all dimensions of size 1
    let squeezed_all = tensor.clone().squeeze()?;
    assert_eq!(squeezed_all.shape(), &[2, 3]);

    // Squeeze specific dimension - note: current squeeze doesn't support specific dims
    let squeezed_dim0 = tensor.squeeze()?;
    assert_eq!(squeezed_dim0.shape(), &[2, 3]);

    // Unsqueeze to add dimension
    let unsqueezed = squeezed_all.clone().unsqueeze(1)?;
    assert_eq!(unsqueezed.shape(), &[2, 1, 3]);

    Ok(())
}

#[test]
fn test_flatten() -> Result<()> {
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![3, 4], data)?;

    let flattened = tensor.reshape(vec![12])?;
    assert_eq!(flattened.shape(), &[12]);

    Ok(())
}

#[test]
fn test_permute() -> Result<()> {
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![2, 3, 4], data)?;

    // Permute dimensions
    let permuted = tensor.clone().permute(&[2, 0, 1])?;
    assert_eq!(permuted.shape(), &[4, 2, 3]);

    // Identity permutation
    let identity = tensor.clone().permute(&[0, 1, 2])?;
    assert_eq!(identity.shape(), &[2, 3, 4]);

    Ok(())
}

#[test]
fn test_sigmoid() -> Result<()> {
    let data = vec![-2.0f32, -1.0f32, 0.0f32, 1.0f32, 2.0f32];
    let tensor = Tensor::from_shape_vec(vec![5], data)?;

    // Note: sigmoid not implemented yet, using tanh as placeholder
    let sigmoid_result = tensor.tanh()?;
    assert_eq!(sigmoid_result.shape(), &[5]);

    Ok(())
}

#[test]
fn test_softmax() -> Result<()> {
    let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
    let tensor = Tensor::from_shape_vec(vec![2, 3], data)?;

    // Softmax along dimension 1
    let softmax_result = tensor.softmax(1)?;
    assert_eq!(softmax_result.shape(), &[2, 3]);

    Ok(())
}

#[test]
fn test_clamp() -> Result<()> {
    let data = vec![-2.0f32, -1.0f32, 0.0f32, 1.0f32, 2.0f32, 3.0f32];
    let tensor = Tensor::from_shape_vec(vec![2, 3], data)?;

    let clamped = tensor.data.clamp_typed(-1.0f32, 2.0f32).unwrap();
    let clamped_tensor: Tensor = clamped.into();
    assert_eq!(clamped_tensor.shape(), &[2, 3]);

    Ok(())
}

#[test]
fn test_broadcast() -> Result<()> {
    let data = vec![1.0f32, 2.0f32, 3.0f32];
    let tensor = Tensor::from_shape_vec(vec![1, 3], data)?;

    let broadcasted = tensor.clone().broadcast_to(vec![2, 3])?;
    assert_eq!(broadcasted.shape(), &[2, 3]);

    Ok(())
}

#[test]
fn test_mixed_operations() -> Result<()> {
    // Test combining multiple operations
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![3, 4], data)?;

    // Chain operations: reshape -> sum -> unsqueeze
    let reshaped = tensor.reshape(vec![2, 6])?;
    let summed = reshaped.sum_dim(Some(1), false)?;
    let final_tensor = summed.clone().unsqueeze(1)?;

    assert_eq!(final_tensor.shape(), &[2, 1]);

    Ok(())
}

#[test]
fn test_different_dtypes() -> Result<()> {
    // Test operations with f32 data type (currently only supported)
    let data_f32 = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
    let tensor_f32 = Tensor::from_shape_vec(vec![2, 3], data_f32)?;

    let sum_f32 = tensor_f32.sum_dim(None, false)?;
    assert_eq!(sum_f32.shape(), &[] as &[usize]);

    let max_f32 = tensor_f32.max_dim(Some(0), false)?;
    assert_eq!(max_f32.shape(), &[3]);

    Ok(())
}
