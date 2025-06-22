//! Comprehensive example demonstrating the USLS tensor slicing API
//!
//! This example showcases the unified `s()` method with native Rust slice syntax,
//! demonstrating various slicing patterns, edge cases, and advanced usage scenarios.
//! The API supports intuitive tuple syntax for ergonomic tensor operations.

use anyhow::Result;
use usls::tensor::Tensor;

fn main() -> Result<()> {
    println!("🦀 USLS Tensor Slicing API Demo 🐍\n");

    // === Basic Tensor Creation ===
    println!("=== 📊 Basic Tensor Creation ===");
    let tensor_3d = Tensor::rand([6, 8, 4], usls::DType::Fp32)?;
    println!("Created 3D tensor: {:?}", tensor_3d.shape());

    let tensor_2d = Tensor::zeros(vec![10, 12]);
    println!("Created 2D tensor: {:?}", tensor_2d.shape());

    let tensor_4d = Tensor::ones(vec![3, 4, 5, 6]);
    println!("Created 4D tensor: {:?}\n", tensor_4d.shape());

    // === Core Slice Syntax Patterns ===
    println!("=== 🎯 Core Slice Syntax Patterns ===");

    // Full slice (..) - selects entire dimension
    let full_slice = tensor_3d.slice((.., .., ..))?;
    println!("Full slice (.., .., ..) -> {:?}", full_slice.shape());

    // Range slice (start..end)
    let range_slice = tensor_3d.slice((1..4, 2..6, ..))?;
    println!("Range slice (1..4, 2..6, ..) -> {:?}", range_slice.shape());

    // Range to (..end)
    let range_to = tensor_3d.slice((.., ..5, ..2))?;
    println!("Range to (.., ..5, ..2) -> {:?}", range_to.shape());

    // Range from (start..)
    let range_from = tensor_3d.slice((2.., 3.., ..))?;
    println!("Range from (2.., 3.., ..) -> {:?}", range_from.shape());

    // Index selection
    let index_slice = tensor_3d.slice((0, .., 2))?;
    println!("Index slice (0, .., 2) -> {:?}\n", index_slice.shape());

    // === Mixed Slicing Patterns ===
    println!("=== 🎨 Mixed Slicing Patterns ===");

    // Combining different slice types
    let mixed1 = tensor_3d.slice((1..5, 0, 1..3))?;
    println!("Mixed (1..5, 0, 1..3) -> {:?}", mixed1.shape());

    let mixed2 = tensor_3d.slice((.., 2..7, 0))?;
    println!("Mixed (.., 2..7, 0) -> {:?}", mixed2.shape());

    let mixed3 = tensor_3d.slice((3.., ..4, 2..))?;
    println!("Mixed (3.., ..4, 2..) -> {:?}\n", mixed3.shape());

    // === Dimensional Reduction Examples ===
    println!("=== 📉 Dimensional Reduction Examples ===");

    // 3D -> 2D (one index)
    let reduce_3d_2d = tensor_3d.slice((2, .., ..))?;
    println!("3D->2D: (2, .., ..) -> {:?}", reduce_3d_2d.shape());

    // 3D -> 1D (two indices)
    let reduce_3d_1d = tensor_3d.slice((1, 3, ..))?;
    println!("3D->1D: (1, 3, ..) -> {:?}", reduce_3d_1d.shape());

    // 3D -> 0D (three indices)
    let reduce_3d_0d = tensor_3d.slice((0, 2, 1))?;
    println!("3D->0D: (0, 2, 1) -> {:?}\n", reduce_3d_0d.shape());

    // === High-Dimensional Tensor Slicing ===
    println!("=== 🚀 High-Dimensional Tensor Slicing ===");

    // 4D tensor complex slicing
    let slice_4d_1 = tensor_4d.slice((0..2, .., 1..4, 2..))?;
    println!("4D slice (0..2, .., 1..4, 2..) -> {:?}", slice_4d_1.shape());

    let slice_4d_2 = tensor_4d.slice((1, 0..3, .., ..3))?;
    println!("4D slice (1, 0..3, .., ..3) -> {:?}", slice_4d_2.shape());

    // 6D tensor demonstration
    let tensor_6d = Tensor::zeros(vec![2, 3, 4, 5, 6, 7]);
    let slice_6d = tensor_6d.slice((0, 1..3, .., 2, .., 1..5))?;
    println!(
        "6D slice (0, 1..3, .., 2, .., 1..5) -> {:?}\n",
        slice_6d.shape()
    );

    // === Chained Slicing Operations ===
    println!("=== ⛓️ Chained Slicing Operations ===");

    let step1 = tensor_3d.slice((1..5, .., ..))?; // [4, 8, 4]
    let step2 = step1.slice((.., 2..6, ..))?; // [4, 4, 4]
    let chained_result = step2.slice((0..2, .., 1..3))?; // [2, 4, 2]
    println!(
        "Chained: (1..5,..,..)->(..2..6,..)->(0..2,..,1..3) -> {:?}",
        chained_result.shape()
    );

    let progressive_slice = tensor_4d
        // .slice((0, .., .., ..))?
        // .slice((.., 1..4, ..))?
        .slice((2, .., 2..5))?;
    println!(
        "Progressive: (0,..,..,..)->(..,1..4,..)->(2,..,2..5) -> {:?}\n",
        progressive_slice.shape()
    );

    // === Edge Cases and Boundary Conditions ===
    println!("=== ⚠️ Edge Cases and Boundary Conditions ===");

    // Empty slices
    let empty_slice = tensor_2d.slice((0..0, ..))?;
    println!("Empty slice (0..0, ..) -> {:?}", empty_slice.shape());

    // Single element slices
    let single_element = tensor_2d.slice((5..6, 3..4))?;
    println!(
        "Single element (5..6, 3..4) -> {:?}",
        single_element.shape()
    );

    // Boundary slicing
    let boundary = tensor_2d.slice((9..10, 11..12))?;
    println!("Boundary slice (9..10, 11..12) -> {:?}\n", boundary.shape());

    // === Equivalent Expression Demonstrations ===
    println!("=== 🔄 Equivalent Expression Demonstrations ===");

    // Different ways to express the same slice
    let equiv1 = tensor_3d.slice((.., ..4, 1..3))?;
    let equiv2 = tensor_3d.slice((.., 0..4, 1..3))?;
    println!(
        "(.., ..4, 1..3) == (.., 0..4, 1..3): {}",
        equiv1.shape() == equiv2.shape()
    );

    let equiv3 = tensor_2d.slice((2.., ..))?;
    let equiv4 = tensor_2d.slice((2..10, 0..12))?;
    println!(
        "(2.., ..) == (2..10, 0..12): {}\n",
        equiv3.shape() == equiv4.shape()
    );

    // === Performance-Critical Patterns ===
    println!("=== ⚡ Performance-Critical Patterns ===");

    let large_tensor = Tensor::zeros(vec![100, 200, 50]);

    // Contiguous slicing (memory efficient)
    let contiguous = large_tensor.slice((10..90, .., ..))?;
    println!(
        "Contiguous slice (10..90, .., ..) -> {:?}",
        contiguous.shape()
    );

    // Strided slicing
    let strided = large_tensor.slice((.., 50..150, 10..40))?;
    println!(
        "Strided slice (.., 50..150, 10..40) -> {:?}",
        strided.shape()
    );

    // Single slice extraction
    let single_extract = large_tensor.slice((42, .., ..))?;
    println!(
        "Single extract (42, .., ..) -> {:?}\n",
        single_extract.shape()
    );

    // === Real-World Use Cases ===
    println!("=== 🌍 Real-World Use Cases ===");

    // Batch processing
    let batch_tensor = Tensor::zeros(vec![32, 224, 224, 3]); // [batch, height, width, channels]
    let single_batch = batch_tensor.slice((0, .., .., ..))?;
    println!("Single batch (0, .., .., ..) -> {:?}", single_batch.shape());

    let batch_subset = batch_tensor.slice((0..8, .., .., ..))?;
    println!(
        "Batch subset (0..8, .., .., ..) -> {:?}",
        batch_subset.shape()
    );

    // Image cropping
    let crop_center = batch_tensor.slice((.., 50..174, 50..174, ..))?;
    println!(
        "Center crop (.., 50..174, 50..174, ..) -> {:?}",
        crop_center.shape()
    );

    // Channel selection
    let rgb_to_r = batch_tensor.slice((.., .., .., 0))?;
    println!("Red channel (.., .., .., 0) -> {:?}\n", rgb_to_r.shape());

    // === Advanced Tensor Manipulations ===
    println!("=== 🎭 Advanced Tensor Manipulations ===");

    // Sequence processing (NLP-like)
    let sequence_tensor = Tensor::zeros(vec![16, 512, 768]); // [batch, seq_len, hidden_dim]
    let truncated_seq = sequence_tensor.slice((.., ..256, ..))?;
    println!(
        "Truncated sequence (.., ..256, ..) -> {:?}",
        truncated_seq.shape()
    );

    // Attention head selection
    let attention_tensor = Tensor::zeros(vec![8, 12, 64, 64]); // [batch, heads, seq, seq]
    let single_head = attention_tensor.slice((.., 0, .., ..))?;
    println!(
        "Single attention head (.., 0, .., ..) -> {:?}",
        single_head.shape()
    );

    // Multi-head subset
    let head_subset = attention_tensor.slice((.., 0..4, .., ..))?;
    println!(
        "Head subset (.., 0..4, .., ..) -> {:?}\n",
        head_subset.shape()
    );

    println!("✨ Tensor slicing demo completed! The s() method provides a unified,");
    println!("   ergonomic interface for all your tensor slicing needs. 🎉");

    Ok(())
}
