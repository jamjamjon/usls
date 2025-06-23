//! Integration tests for the reversed_axes() method
//!
//! These tests verify the correctness of the reversed_axes() method
//! for Tensor, TensorView, and TensorViewMut types.

use anyhow::Result;
use usls::tensor::Tensor;
use usls::DType;

#[test]
fn test_tensor_reversed_axes_basic() -> Result<()> {
    // Test basic functionality
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![2, 3, 4], data)?;
    let reversed = tensor.reversed_axes()?;

    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(reversed.shape(), &[4, 3, 2]);

    Ok(())
}

#[test]
fn test_tensor_view_reversed_axes() -> Result<()> {
    let data: Vec<f32> = (0..60).map(|x| x as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![3, 4, 5], data)?;
    let view = tensor.view();
    let reversed = view.reversed_axes()?;

    assert_eq!(view.shape(), &[3, 4, 5]);
    assert_eq!(reversed.shape(), &[5, 4, 3]);

    Ok(())
}

#[test]
fn test_tensor_view_mut_reversed_axes() -> Result<()> {
    let data: Vec<f32> = (0..120).map(|x| x as f32).collect();
    let mut tensor = Tensor::from_shape_vec(vec![2, 3, 4, 5], data)?;
    let reversed = {
        let view_mut = tensor.view_mut();
        assert_eq!(view_mut.shape(), &[2, 3, 4, 5]);
        view_mut.reversed_axes()?
    };

    assert_eq!(reversed.shape(), &[5, 4, 3, 2]);

    Ok(())
}

#[test]
fn test_reversed_axes_different_dtypes() -> Result<()> {
    let dtypes = vec![
        DType::Fp32,
        DType::Fp64,
        DType::Int8,
        DType::Int16,
        DType::Int32,
        DType::Int64,
        DType::Uint8,
        DType::Uint16,
        DType::Uint32,
        DType::Uint64,
    ];

    for dtype in dtypes {
        let tensor = Tensor::zeros_with_dtype(vec![2, 3, 4], dtype);
        let reversed = tensor.reversed_axes()?;

        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(reversed.shape(), &[4, 3, 2]);
        assert_eq!(tensor.dtype(), reversed.dtype());
    }

    Ok(())
}

#[test]
fn test_reversed_axes_1d() -> Result<()> {
    let data: Vec<f32> = (0..5).map(|x| x as f32).collect();
    let tensor = Tensor::from_vec(data);
    let reversed = tensor.reversed_axes()?;

    assert_eq!(tensor.shape(), &[5]);
    assert_eq!(reversed.shape(), &[5]);

    Ok(())
}

#[test]
fn test_reversed_axes_2d() -> Result<()> {
    let tensor = Tensor::zeros_with_dtype(vec![3, 4], DType::Fp32);
    let reversed = tensor.reversed_axes()?;

    assert_eq!(tensor.shape(), &[3, 4]);
    assert_eq!(reversed.shape(), &[4, 3]);

    Ok(())
}

#[test]
fn test_reversed_axes_5d() -> Result<()> {
    let tensor = Tensor::zeros_with_dtype(vec![1, 2, 3, 4, 5], DType::Fp32);
    let reversed = tensor.reversed_axes()?;

    assert_eq!(tensor.shape(), &[1, 2, 3, 4, 5]);
    assert_eq!(reversed.shape(), &[5, 4, 3, 2, 1]);

    Ok(())
}

#[test]
fn test_reversed_axes_data_integrity() -> Result<()> {
    // Create a simple 2x3 tensor with known values
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_shape_vec(vec![2, 3], data)?;
    let reversed = tensor.reversed_axes()?;

    // Original tensor: [[1, 2, 3], [4, 5, 6]] (2x3)
    // Reversed tensor: [[1, 4], [2, 5], [3, 6]] (3x2)

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(reversed.shape(), &[3, 2]);

    // Check that the data is correctly transposed
    let original_data = tensor.to_vec::<f32>()?;
    let reversed_data = reversed.to_vec::<f32>()?;

    assert_eq!(original_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(reversed_data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    Ok(())
}

#[test]
fn test_reversed_axes_double_reverse() -> Result<()> {
    // Reversing twice should give back the original shape
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![2, 3, 4], data)?;
    let reversed_once = tensor.reversed_axes()?;
    let reversed_twice = reversed_once.reversed_axes()?;

    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(reversed_once.shape(), &[4, 3, 2]);
    assert_eq!(reversed_twice.shape(), &[2, 3, 4]);

    // Data should be the same after double reverse
    let original_data = tensor.to_vec::<f32>()?;
    let double_reversed_data = reversed_twice.to_vec::<f32>()?;
    assert_eq!(original_data, double_reversed_data);

    Ok(())
}

#[test]
fn test_reversed_axes_with_slicing() -> Result<()> {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![2, 3, 4], data)?;

    // Create a view with slicing
    let view = tensor.slice(&[0..1, 1..3, 0..2])?;
    assert_eq!(view.shape(), &[1, 2, 2]);

    // Reverse the axes of the sliced view
    let reversed = view.reversed_axes()?;
    assert_eq!(reversed.shape(), &[2, 2, 1]);

    Ok(())
}

#[test]
fn test_reversed_axes_f16_bf16() -> Result<()> {
    // Test with half precision types
    let tensor_f16 = Tensor::zeros_with_dtype(vec![2, 3], DType::Fp16);
    let reversed_f16 = tensor_f16.reversed_axes()?;
    assert_eq!(reversed_f16.shape(), &[3, 2]);

    let tensor_bf16 = Tensor::zeros_with_dtype(vec![2, 3], DType::Bf16);
    let reversed_bf16 = tensor_bf16.reversed_axes()?;
    assert_eq!(reversed_bf16.shape(), &[3, 2]);

    Ok(())
}
