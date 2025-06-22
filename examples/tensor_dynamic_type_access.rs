//! Example demonstrating dynamic type access for tensors
//!
//! This example shows how to use the new `as_array_typed<T>()` and `get_element<T>()`
//! methods to safely access tensor data of different types without manual conversion.

use anyhow::Result;
use usls::Tensor;

fn main() -> Result<()> {
    println!("🦀 Tensor Dynamic Type Access Demo");

    // Create tensors of different types
    let f32_tensor = Tensor::from_shape_vec([4], vec![1.0f32, 2.0, 3.0, 4.0])?;
    let i64_tensor = Tensor::from_shape_vec([4], vec![10i64, 20, 30, 40])?;
    let bool_tensor = Tensor::from_shape_vec([4], vec![true, false, true, false])?;

    println!("\n📊 Tensor Information:");
    println!(
        "F32 tensor: shape={:?}, dtype={:?}",
        f32_tensor.shape(),
        f32_tensor.dtype()
    );
    println!(
        "I64 tensor: shape={:?}, dtype={:?}",
        i64_tensor.shape(),
        i64_tensor.dtype()
    );
    println!(
        "Bool tensor: shape={:?}, dtype={:?}",
        bool_tensor.shape(),
        bool_tensor.dtype()
    );

    // Method 1: Using get_element<T>() for single element access
    println!("\n🎯 Method 1: get_element<T>() - Single Element Access");

    // Access F32 tensor elements
    for i in 0..4 {
        let val: f32 = f32_tensor.get_element(i)?;
        println!("f32_tensor[{}] = {}", i, val);
    }

    // Access I64 tensor elements (with automatic type conversion)
    for i in 0..4 {
        let val_as_i64: i64 = i64_tensor.get_element(i)?;
        let val_as_f32: f32 = i64_tensor.get_element(i)?; // Auto conversion
        println!(
            "i64_tensor[{}] = {} (as i64), {} (as f32)",
            i, val_as_i64, val_as_f32
        );
    }

    // Access Bool tensor elements
    for i in 0..4 {
        let val_as_bool: bool = bool_tensor.get_element(i)?;
        let val_as_f32: f32 = bool_tensor.get_element(i)?; // Auto conversion
        println!(
            "bool_tensor[{}] = {} (as bool), {} (as f32)",
            i, val_as_bool, val_as_f32
        );
    }

    // Method 2: Using as_array_typed<T>() for direct array access
    println!("\n🔗 Method 2: as_array_typed<T>() - Direct Array Access");

    // Access F32 tensor as typed array
    let f32_array = f32_tensor.as_array_typed::<f32>()?;
    println!("F32 array slice: {:?}", f32_array.as_slice().unwrap());

    // Access I64 tensor as typed array
    let i64_array = i64_tensor.as_array_typed::<i64>()?;
    println!("I64 array slice: {:?}", i64_array.as_slice().unwrap());

    // Access Bool tensor as typed array
    let bool_array = bool_tensor.as_array_typed::<bool>()?;
    println!("Bool array slice: {:?}", bool_array.as_slice().unwrap());

    // Method 3: Error handling for type mismatches
    println!("\n⚠️  Method 3: Type Safety Demonstration");

    // This will fail gracefully
    match f32_tensor.as_array_typed::<i64>() {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => println!("Expected error when accessing F32 tensor as I64: {}", e),
    }

    // Method 4: Practical usage pattern (similar to RTDETR fix)
    println!("\n🎯 Method 4: Practical Usage Pattern");

    let labels = Tensor::from_shape_vec([5], vec![0i64, 1, 2, 1, 0])?;
    let scores = Tensor::from_shape_vec([5], vec![0.9f32, 0.8, 0.7, 0.6, 0.5])?;

    println!("Processing detection results:");
    for i in 0..5 {
        let class_id: i64 = labels.get_element(i)?;
        let confidence: f32 = scores.get_element(i)?;
        println!(
            "Detection {}: class_id={}, confidence={:.2}",
            i, class_id, confidence
        );
    }

    println!("\n✅ All dynamic type access methods work correctly!");
    Ok(())
}
