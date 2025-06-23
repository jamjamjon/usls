//! Integration tests for tensor random functions

use anyhow::Result;
use usls::Tensor;

#[test]
fn test_rand_f32() -> Result<()> {
    let tensor = Tensor::rand(0.0f32, 1.0f32, vec![10, 10])?;
    assert_eq!(tensor.shape(), &[10, 10]);
    assert_eq!(tensor.dtype(), usls::DType::Fp32);

    // Check values are in range [0.0, 1.0)
    let data: &[f32] = tensor.as_slice::<f32>().unwrap();
    for &val in data {
        assert!(
            (0.0..1.0).contains(&val),
            "Value {} out of range [0.0, 1.0)",
            val
        );
    }
    Ok(())
}

#[test]
fn test_rand_f64() -> Result<()> {
    let tensor = Tensor::rand(-1.0f64, 1.0f64, vec![5, 5])?;
    assert_eq!(tensor.shape(), &[5, 5]);
    assert_eq!(tensor.dtype(), usls::DType::Fp64);

    // Check values are in range [-1.0, 1.0)
    let data: &[f64] = tensor.as_slice::<f64>().unwrap();
    for &val in data {
        assert!(
            (-1.0..1.0).contains(&val),
            "Value {} out of range [-1.0, 1.0)",
            val
        );
    }
    Ok(())
}

#[test]
fn test_rand_i32() -> Result<()> {
    let tensor = Tensor::rand(0i32, 100i32, vec![3, 3])?;
    assert_eq!(tensor.shape(), &[3, 3]);
    assert_eq!(tensor.dtype(), usls::DType::Int32);

    // Check values are in range [0, 100)
    let data: &[i32] = tensor.as_slice::<i32>().unwrap();
    for &val in data {
        assert!(
            (0..100).contains(&val),
            "Value {} out of range [0, 100)",
            val
        );
    }
    Ok(())
}

#[test]
fn test_rand_u8() -> Result<()> {
    let tensor = Tensor::rand(50u8, 200u8, vec![4, 4])?;
    assert_eq!(tensor.shape(), &[4, 4]);
    assert_eq!(tensor.dtype(), usls::DType::Uint8);

    // Check values are in range [50, 200)
    let data: &[u8] = tensor.as_slice::<u8>().unwrap();
    for &val in data {
        assert!(
            (50..200).contains(&val),
            "Value {} out of range [50, 200)",
            val
        );
    }
    Ok(())
}

#[test]
fn test_randn_f32() -> Result<()> {
    let tensor = Tensor::randn::<f32, _>(vec![100, 100])?;
    assert_eq!(tensor.shape(), &[100, 100]);
    assert_eq!(tensor.dtype(), usls::DType::Fp32);

    // Check that we have reasonable normal distribution properties
    let data: &[f32] = tensor.as_slice::<f32>().unwrap();
    let mean: f32 = data.iter().copied().sum::<f32>() / data.len() as f32;
    let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;

    // For standard normal, mean should be close to 0, variance close to 1
    assert!(mean.abs() < 0.1, "Mean {} too far from 0", mean);
    assert!(
        (variance - 1.0).abs() < 0.1,
        "Variance {} too far from 1",
        variance
    );
    Ok(())
}

#[test]
fn test_randn_f64() -> Result<()> {
    let tensor = Tensor::randn::<f64, _>(vec![50, 50])?;
    assert_eq!(tensor.shape(), &[50, 50]);
    assert_eq!(tensor.dtype(), usls::DType::Fp64);

    // Check that we have reasonable normal distribution properties
    let data: &[f64] = tensor.as_slice::<f64>().unwrap();
    let mean: f64 = data.iter().copied().sum::<f64>() / data.len() as f64;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

    // For standard normal, mean should be close to 0, variance close to 1
    assert!(mean.abs() < 0.1, "Mean {} too far from 0", mean);
    assert!(
        (variance - 1.0).abs() < 0.1,
        "Variance {} too far from 1",
        variance
    );
    Ok(())
}

#[test]
fn test_rand_different_shapes() -> Result<()> {
    // Test 1D
    let tensor1d = Tensor::rand(0.0f32, 1.0f32, 100)?;
    assert_eq!(tensor1d.shape(), &[100]);

    // Test 3D
    let tensor3d = Tensor::rand(0.0f32, 1.0f32, vec![2, 3, 4])?;
    assert_eq!(tensor3d.shape(), &[2, 3, 4]);

    // Test 4D
    let tensor4d = Tensor::rand(0.0f32, 1.0f32, vec![2, 2, 2, 2])?;
    assert_eq!(tensor4d.shape(), &[2, 2, 2, 2]);

    Ok(())
}

#[test]
fn test_randn_different_shapes() -> Result<()> {
    // Test 1D
    let tensor1d = Tensor::randn::<f32, _>(50)?;
    assert_eq!(tensor1d.shape(), &[50]);

    // Test 3D
    let tensor3d = Tensor::randn::<f32, _>(vec![2, 3, 4])?;
    assert_eq!(tensor3d.shape(), &[2, 3, 4]);

    // Test 4D
    let tensor4d = Tensor::randn::<f32, _>(vec![2, 2, 2, 2])?;
    assert_eq!(tensor4d.shape(), &[2, 2, 2, 2]);

    Ok(())
}

#[test]
fn test_rand_edge_cases() -> Result<()> {
    // Test with very small range
    let tensor = Tensor::rand(0.5f32, 0.6f32, vec![10, 10])?;
    let data: &[f32] = tensor.as_slice::<f32>().unwrap();
    for &val in data {
        assert!(
            (0.5..0.6).contains(&val),
            "Value {} out of range [0.5, 0.6)",
            val
        );
    }

    // Test with negative range
    let tensor = Tensor::rand(-100i32, -50i32, vec![5, 5])?;
    let data: &[i32] = tensor.as_slice::<i32>().unwrap();
    for &val in data {
        assert!(
            (-100..-50).contains(&val),
            "Value {} out of range [-100, -50)",
            val
        );
    }

    Ok(())
}
