//! Tensor View Demo
//! TensorView and TensorViewMut Demo
//!
//! This example demonstrates the zero-copy view operations for tensors,
//! showcasing performance-critical patterns like KV cache operations,
//! image processing, and memory-efficient tensor manipulations.

use anyhow::Result;
use usls::tensor::Tensor;

fn main() -> Result<()> {
    println!("🦀 USLS TensorView & TensorViewMut Demo 🚀\n");

    // === Basic TensorView Operations ===
    println!("=== 📊 Basic TensorView Operations ===");
    basic_tensor_view_demo()?;

    // === TensorViewMut Operations ===
    println!("\n=== 🔧 TensorViewMut Operations ===");
    tensor_view_mut_demo()?;

    // === Image Processing with Views ===
    println!("\n=== 🖼️ Image Processing with Views ===");
    image_processing_demo()?;

    // === KV Cache Operations ===
    println!("\n=== 🧠 KV Cache Operations ===");
    kv_cache_demo()?;

    // === Sequence Processing ===
    println!("\n=== 📝 Sequence Processing ===");
    sequence_processing_demo()?;

    // === Performance Patterns ===
    println!("\n=== ⚡ Performance Patterns ===");
    performance_patterns_demo()?;

    // === Advanced View Operations ===
    println!("\n=== 🎭 Advanced View Operations ===");
    advanced_view_operations()?;

    println!("\n✨ TensorView demo completed! Zero-copy views enable");
    println!("   efficient memory usage and high-performance operations. 🎉");

    Ok(())
}

fn basic_tensor_view_demo() -> Result<()> {
    let tensor = Tensor::rand([6, 8, 4], usls::DType::Fp32)?;
    println!("Original tensor shape: {:?}", tensor.shape());

    // Method 1: Create view of entire tensor using .view()
    let full_view = tensor.view();
    println!("Full view shape: {:?}", full_view.shape());
    println!(
        "Full view matches original: {}",
        full_view.shape() == tensor.shape()
    );

    // Method 2: Create immutable view using slice method
    let view = tensor.slice((1..5, 2..6, 0..4))?;

    println!("Partial view shape: {:?}", view.shape());
    println!("View dtype: {:?}", view.dtype());
    println!("View ndim: {}", view.ndim());
    println!("View len: {}", view.len());
    println!("View is_empty: {}", view.is_empty());

    // Convert view to owned tensor
    let owned = view.to_owned()?;
    println!("Converted to owned tensor: {:?}", owned.shape());

    // Create sub-view
    let sub_view = view.slice((1..3, 1..3, 1..3))?;
    println!("Sub-view shape: {:?}", sub_view.shape());

    Ok(())
}

fn tensor_view_mut_demo() -> Result<()> {
    let mut tensor = Tensor::zeros(vec![4, 6, 3]);
    println!("Mutable tensor shape: {:?}", tensor.shape());

    // Method 1: Create mutable view of entire tensor using .view_mut()
    {
        let mut full_view = tensor.view_mut();
        println!("Full mutable view shape: {:?}", full_view.shape());

        // Fill the entire tensor through the view
        full_view.fill(1.5)?;
        println!("Filled entire tensor with 1.5 through view_mut()");
    }

    // Method 2: Create mutable view using slice_mut method
    let mut view = tensor.slice_mut((1..3, 2..5, 0..3))?;

    println!("Partial mutable view shape: {:?}", view.shape());
    println!("Mutable view dtype: {:?}", view.dtype());

    // Convert mutable view to owned tensor
    let owned = view.to_owned()?;
    println!("Converted to owned tensor: {:?}", owned.shape());

    // Create mutable sub-slice
    let sub_slice = view.slice_mut((0..2, 1..3, 1..2))?;
    println!("Mutable sub-slice shape: {:?}", sub_slice.shape());

    Ok(())
}

fn image_processing_demo() -> Result<()> {
    // Simulate batch image data [batch, height, width, channels]
    let batch_images = Tensor::rand([8, 224, 224, 3], usls::DType::Fp32)?;
    println!("Batch images shape: {:?}", batch_images.shape());

    // Single image view
    let single_image = batch_images.slice((0..1, 0..224, 0..224, 0..3))?;
    println!("Single image view: {:?}", single_image.shape());

    // Center crop
    let cropped = batch_images.slice((0..8, 50..174, 50..174, 0..3))?;
    println!("Center cropped: {:?}", cropped.shape());

    // Channel selection (Red channel only)
    let red_channel = batch_images.slice((0..8, 0..224, 0..224, 0..1))?;
    println!("Red channel only: {:?}", red_channel.shape());

    // Batch subset
    let batch_subset = batch_images.slice((0..4, 0..224, 0..224, 0..3))?;
    println!("Batch subset: {:?}", batch_subset.shape());

    // Corner crop
    let corner_crop = batch_images.slice((0..8, 0..112, 0..112, 0..3))?;
    println!("Corner crop: {:?}", corner_crop.shape());

    Ok(())
}

fn kv_cache_demo() -> Result<()> {
    // Simulate KV cache [batch, num_heads, seq_len, head_dim]
    let kv_cache = Tensor::zeros(vec![4, 12, 512, 64]);
    println!("KV cache shape: {:?}", kv_cache.shape());

    // Single head cache view
    let single_head = kv_cache.slice((0..4, 0..1, 0..512, 0..64))?;
    println!("Single head cache: {:?}", single_head.shape());

    // Multi-head subset
    let multi_head = kv_cache.slice((0..4, 0..4, 0..512, 0..64))?;
    println!("Multi-head subset: {:?}", multi_head.shape());

    // Sequence window
    let seq_window = kv_cache.slice((0..4, 0..12, 100..200, 0..64))?;
    println!("Sequence window: {:?}", seq_window.shape());

    // Single batch cache
    let single_batch = kv_cache.slice((0..1, 0..12, 0..512, 0..64))?;
    println!("Single batch cache: {:?}", single_batch.shape());

    // Head dimension subset
    let head_dim_subset = kv_cache.slice((0..4, 0..12, 0..512, 0..32))?;
    println!("Head dim subset: {:?}", head_dim_subset.shape());

    Ok(())
}

fn sequence_processing_demo() -> Result<()> {
    // Simulate sequence data [batch, seq_len, hidden_dim]
    let sequence = Tensor::rand([16, 512, 768], usls::DType::Fp32)?;
    println!("Sequence shape: {:?}", sequence.shape());

    // Truncate sequence
    let truncated = sequence.slice((0..16, 0..256, 0..768))?;
    println!("Truncated sequence: {:?}", truncated.shape());

    // Batch subset
    let batch_subset = sequence.slice((0..8, 0..512, 0..768))?;
    println!("Batch subset: {:?}", batch_subset.shape());

    // Feature dimension subset
    let feature_subset = sequence.slice((0..16, 0..512, 0..384))?;
    println!("Feature subset: {:?}", feature_subset.shape());

    // Sequence window
    let sequence_window = sequence.slice((0..16, 128..384, 0..768))?;
    println!("Sequence window: {:?}", sequence_window.shape());

    // Single sequence
    let single_sequence = sequence.slice((0..1, 0..512, 0..768))?;
    println!("Single sequence: {:?}", single_sequence.shape());

    Ok(())
}

fn performance_patterns_demo() -> Result<()> {
    let large_tensor = Tensor::zeros(vec![1000, 1000, 100]);
    println!("Large tensor shape: {:?}", large_tensor.shape());

    // Contiguous slicing (memory efficient)
    let contiguous = large_tensor.slice((100..900, 0..1000, 0..100))?;
    println!("Contiguous slice: {:?}", contiguous.shape());

    // Strided access
    let strided = large_tensor.slice((0..1000, 100..900, 10..90))?;
    println!("Strided slice: {:?}", strided.shape());

    // Single slice extraction
    let single_slice = large_tensor.slice((500..501, 0..1000, 0..100))?;
    println!("Single slice: {:?}", single_slice.shape());

    // Batch processing simulation
    let batch_size = 100;
    for i in (0..1000).step_by(batch_size) {
        let end = (i + batch_size).min(1000);
        let batch = large_tensor.slice((i..end, 0..1000, 0..100))?;

        if i < 300 {
            // Only print first few batches
            println!("Batch {}-{}: {:?}", i, end - 1, batch.shape());
        }
    }

    Ok(())
}

fn advanced_view_operations() -> Result<()> {
    let tensor = Tensor::rand([8, 8, 8, 8], usls::DType::Fp32)?;
    println!("4D tensor shape: {:?}", tensor.shape());

    // Chained view operations
    let view1 = tensor.slice((1..7, 0..8, 0..8, 0..8))?;
    println!("First view: {:?}", view1.shape());

    let view2 = view1.slice((0..4, 2..6, 0..8, 0..8))?;
    println!("Second view: {:?}", view2.shape());

    let final_view = view2.slice((0..2, 0..4, 1..7, 2..6))?;
    println!("Final chained view: {:?}", final_view.shape());

    // Complex indexing patterns
    let complex_view = tensor.slice((0..4, 1..7, 2..6, 3..5))?;
    println!("Complex view: {:?}", complex_view.shape());

    // Edge case: single element views
    let single_element = tensor.slice((3..4, 4..5, 5..6, 6..7))?;
    println!("Single element view: {:?}", single_element.shape());

    // Dimension reduction through indexing
    let reduced = tensor.slice((0..8, 3..4, 0..8, 0..8))?;
    println!("Dimension reduced: {:?}", reduced.shape());

    Ok(())
}
