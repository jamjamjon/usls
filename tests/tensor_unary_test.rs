use anyhow::Result;
use usls::tensor::Tensor;

#[test]
fn test_abs_operations() -> Result<()> {
    // Test abs with f32 - using zeros and then testing the operation
    let tensor_f32 = Tensor::zeros(vec![4]);
    let abs_result = tensor_f32.abs()?;

    // Verify shape is preserved
    assert_eq!(abs_result.shape(), &[4]);

    // Test abs with different tensor sizes
    let tensor_2d = Tensor::zeros(vec![2, 3]);
    let abs_result_2d = tensor_2d.abs()?;

    assert_eq!(abs_result_2d.shape(), &[2, 3]);

    // Test abs with 3D tensor
    let tensor_3d = Tensor::zeros(vec![2, 2, 2]);
    let abs_result_3d = tensor_3d.abs()?;

    assert_eq!(abs_result_3d.shape(), &[2, 2, 2]);

    Ok(())
}

#[test]
fn test_sqrt_operations() -> Result<()> {
    // Test sqrt with different tensor shapes
    let tensor = Tensor::zeros(vec![4]);
    let sqrt_result = tensor.sqrt()?;

    assert_eq!(sqrt_result.shape(), &[4]);

    // Test sqrt with 2D tensor
    let tensor_2d = Tensor::zeros(vec![3, 2]);
    let sqrt_result_2d = tensor_2d.sqrt()?;

    assert_eq!(sqrt_result_2d.shape(), &[3, 2]);

    Ok(())
}

#[test]
fn test_exp_operations() -> Result<()> {
    // Test exp with different tensor shapes
    let tensor = Tensor::zeros(vec![3]);
    let exp_result = tensor.exp()?;

    assert_eq!(exp_result.shape(), &[3]);

    // Test exp with 2D tensor
    let tensor_2d = Tensor::zeros(vec![2, 4]);
    let exp_result_2d = tensor_2d.exp()?;

    assert_eq!(exp_result_2d.shape(), &[2, 4]);

    Ok(())
}

#[test]
fn test_log_operations() -> Result<()> {
    // Test log (natural logarithm) with different shapes
    let tensor = Tensor::zeros(vec![3]);
    let log_result = tensor.log()?;

    assert_eq!(log_result.shape(), &[3]);

    // Test log with 2D tensor
    let tensor_2d = Tensor::zeros(vec![2, 3]);
    let log_result_2d = tensor_2d.log()?;

    assert_eq!(log_result_2d.shape(), &[2, 3]);

    Ok(())
}

#[test]
fn test_sin_operations() -> Result<()> {
    // Test sin with different tensor shapes
    let tensor = Tensor::zeros(vec![3]);
    let sin_result = tensor.sin()?;

    assert_eq!(sin_result.shape(), &[3]);

    // Test sin with 2D tensor
    let tensor_2d = Tensor::zeros(vec![2, 3]);
    let sin_result_2d = tensor_2d.sin()?;

    assert_eq!(sin_result_2d.shape(), &[2, 3]);

    Ok(())
}

#[test]
fn test_cos_operations() -> Result<()> {
    // Test cos with different tensor shapes
    let tensor = Tensor::zeros(vec![3]);
    let cos_result = tensor.cos()?;

    assert_eq!(cos_result.shape(), &[3]);

    // Test cos with 2D tensor
    let tensor_2d = Tensor::zeros(vec![2, 4]);
    let cos_result_2d = tensor_2d.cos()?;

    assert_eq!(cos_result_2d.shape(), &[2, 4]);

    Ok(())
}

#[test]
fn test_tanh_operations() -> Result<()> {
    // Test tanh with different tensor shapes
    let tensor = Tensor::zeros(vec![2]);
    let tanh_result = tensor.tanh()?;

    assert_eq!(tanh_result.shape(), &[2]);

    // Test tanh with 3D tensor
    let tensor_3d = Tensor::zeros(vec![2, 2, 2]);
    let tanh_result_3d = tensor_3d.tanh()?;

    assert_eq!(tanh_result_3d.shape(), &[2, 2, 2]);

    Ok(())
}

#[test]
fn test_neg_operations() -> Result<()> {
    // Test neg with different tensor shapes
    let tensor = Tensor::zeros(vec![4]);
    let neg_result = tensor.neg()?;

    assert_eq!(neg_result.shape(), &[4]);

    // Test neg with 2D tensor
    let tensor_2d = Tensor::zeros(vec![3, 2]);
    let neg_result_2d = tensor_2d.neg()?;

    assert_eq!(neg_result_2d.shape(), &[3, 2]);

    Ok(())
}

#[test]
fn test_chained_unary_operations() -> Result<()> {
    // Test chaining multiple unary operations
    let tensor = Tensor::zeros(vec![3]);

    // Chain: abs -> sqrt -> exp
    let result = tensor.abs()?.sqrt()?.exp()?;

    assert_eq!(result.shape(), &[3]);

    Ok(())
}

#[test]
fn test_multidimensional_unary_operations() -> Result<()> {
    // Test unary operations on multi-dimensional tensors
    let tensor_2d = Tensor::zeros(vec![2, 3]);

    let abs_result = tensor_2d.abs()?;
    assert_eq!(abs_result.shape(), &[2, 3]);

    let exp_result = abs_result.exp()?;
    assert_eq!(exp_result.shape(), &[2, 3]);

    // Test 3D tensor
    let tensor_3d = Tensor::zeros(vec![2, 3, 4]);
    let sin_result = tensor_3d.sin()?;
    assert_eq!(sin_result.shape(), &[2, 3, 4]);

    Ok(())
}

#[test]
fn test_unary_operations_preserve_shape() -> Result<()> {
    // Test that unary operations preserve tensor shape
    let tensor_1d = Tensor::zeros(vec![5]);
    let abs_result = tensor_1d.abs()?;

    // Shape should be preserved
    assert_eq!(abs_result.shape(), &[5]);

    // Test with 4D tensor
    let tensor_4d = Tensor::zeros(vec![2, 3, 4, 5]);
    let sqrt_result = tensor_4d.sqrt()?;

    assert_eq!(sqrt_result.shape(), &[2, 3, 4, 5]);

    Ok(())
}
