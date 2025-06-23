//! Demonstration of the improved Tensor::rand API with automatic type inference

use anyhow::Result;
use usls::Tensor;

fn main() -> Result<()> {
    // f32 type inferred from literal values
    let f32_tensor = Tensor::rand(0.0f32, 1.0, [1, 4])?;
    println!("# f32 tensor(0.0-1.0): {:?}", f32_tensor);

    // f64 type inferred from literal values
    let f64_tensor = Tensor::rand(1.0f64, 10.0f64, [3, 1])?;
    println!("# f64 tensor(1.0-10.0): {:?}", f64_tensor);

    // i32 type inferred from literal values
    let i32_tensor = Tensor::rand(-50i32, 100i32, [2, 2])?;
    println!("# i32 tensor(-50-100): {:?}", i32_tensor);

    // u8 type inferred from literal values
    let u8_tensor = Tensor::rand(50u8, 200u8, [2, 1])?;
    println!("# u8 tensor(50-200): {:?}", u8_tensor);

    // 1D tensor
    let tensor_1d = Tensor::rand(0.0f32, 1.0f32, 10)?;
    println!("# 1D tensor: {:?}", tensor_1d.shape());

    // 2D tensor
    let tensor_2d = Tensor::rand(0.0f32, 1.0f32, [5, 8])?;
    println!("# 2D tensor: {:?}", tensor_2d.shape());

    // 3D tensor
    let tensor_3d = Tensor::rand(0.0f32, 1.0f32, [2, 3, 4])?;
    println!("# 3D tensor: {:?}", tensor_3d.shape());

    // 4D tensor (common for image batches)
    let tensor_4d = Tensor::rand(0.0f32, 1.0f32, [8, 224, 224, 3])?;
    println!("# 4D tensor: {:?}", tensor_4d.shape());

    Ok(())
}
