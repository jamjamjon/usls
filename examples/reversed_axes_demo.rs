//! Demo for the reversed_axes() method
//!
//! This example demonstrates how to use the reversed_axes() method
//! on Tensor, TensorView, and TensorViewMut types.

use anyhow::Result;
use usls::tensor::Tensor;
use usls::DType;

fn main() -> Result<()> {
    println!("🦀 Tensor reversed_axes() Demo 🦀\n");

    // Demo 1: Basic Tensor reversed_axes
    println!("📊 Demo 1: Basic Tensor reversed_axes");
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![2, 3, 4], data)?;
    println!("Original tensor shape: {:?}", tensor.shape());

    let reversed = tensor.reversed_axes()?;
    println!("Reversed tensor shape: {:?}", reversed.shape());
    println!("✅ Tensor reversed_axes works!\n");

    // Demo 2: TensorView reversed_axes
    println!("👀 Demo 2: TensorView reversed_axes");
    let view = tensor.view();
    println!("Original view shape: {:?}", view.shape());

    let reversed_from_view = view.reversed_axes()?;
    println!("Reversed from view shape: {:?}", reversed_from_view.shape());
    println!("✅ TensorView reversed_axes works!\n");

    // Demo 3: TensorViewMut reversed_axes
    println!("✏️ Demo 3: TensorViewMut reversed_axes");
    let data_mut: Vec<f32> = (0..60).map(|x| x as f32).collect();
    let mut tensor_mut = Tensor::from_shape_vec(vec![3, 4, 5], data_mut)?;
    {
        let view_mut = tensor_mut.view_mut();
        println!("Original mutable view shape: {:?}", view_mut.shape());

        let reversed_from_mut_view = view_mut.reversed_axes()?;
        println!(
            "Reversed from mutable view shape: {:?}",
            reversed_from_mut_view.shape()
        );
    }
    println!("✅ TensorViewMut reversed_axes works!\n");

    // Demo 4: Different data types
    println!("🔢 Demo 4: Different data types");
    let dtypes = vec![
        (DType::Fp32, "Fp32"),
        (DType::Fp64, "Fp64"),
        (DType::Int32, "Int32"),
        (DType::Uint8, "Uint8"),
    ];

    for (dtype, name) in dtypes {
        let tensor = Tensor::zeros_with_dtype(vec![2, 3], dtype);
        let reversed = tensor.reversed_axes()?;
        println!(
            "  {} tensor: {:?} -> {:?}",
            name,
            tensor.shape(),
            reversed.shape()
        );
    }
    println!("✅ All data types work!\n");

    // Demo 5: Edge cases
    println!("🎯 Demo 5: Edge cases");

    // 1D tensor
    let data_1d: Vec<f32> = (0..5).map(|x| x as f32).collect();
    let tensor_1d = Tensor::from_vec(data_1d);
    let reversed_1d = tensor_1d.reversed_axes()?;
    println!(
        "  1D tensor: {:?} -> {:?}",
        tensor_1d.shape(),
        reversed_1d.shape()
    );

    // 2D tensor
    let tensor_2d = Tensor::zeros_with_dtype(vec![3, 4], DType::Fp32);
    let reversed_2d = tensor_2d.reversed_axes()?;
    println!(
        "  2D tensor: {:?} -> {:?}",
        tensor_2d.shape(),
        reversed_2d.shape()
    );

    // 5D tensor
    let tensor_5d = Tensor::zeros_with_dtype(vec![1, 2, 3, 4, 5], DType::Fp32);
    let reversed_5d = tensor_5d.reversed_axes()?;
    println!(
        "  5D tensor: {:?} -> {:?}",
        tensor_5d.shape(),
        reversed_5d.shape()
    );

    println!("✅ Edge cases work!\n");

    println!("🎉 All demos completed successfully! 🎉");
    Ok(())
}
