//! Example demonstrating dynamic type access for tensors
//!
//! This example shows how to use the new `as_array_typed<T>()` and `get_element<T>()`
//! methods to safely access tensor data of different types without manual conversion.

use anyhow::Result;
use usls::Tensor;

fn main() -> Result<()> {
    // Create tensors of different types
    let f64_tensor = Tensor::from_shape_vec([4], vec![1.0f64, 2.0, 3.0, 4.0])?;
    let f32_tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
    let i64_tensor = Tensor::from_slice(&[10i64, 20, 30, 40]);
    let bool_tensor = Tensor::from(vec![true, false, true, false]);

    println!("f64_tensor: {:?}", f64_tensor);
    println!("f32_tensor: {:?}", f32_tensor);
    println!("i64_tensor: {:?}", i64_tensor);
    println!("bool_tensor: {:?}", bool_tensor);

    let f32_tensor_rand = Tensor::rand(0.0f32, 1.0f32, [4])?;
    println!("f32_tensor_rand: {:?}", f32_tensor_rand);

    // let z = f32_tensor * f32_tensor_rand;
    // println!("z: {:?}", z);

    // println!("f32_tensor: {:?}", f32_tensor);
    // println!("f32_tensor_rand: {:?}", f32_tensor_rand);

    let z = f32_tensor * 1.;
    println!("z: {:?}", z);

    Ok(())
}
