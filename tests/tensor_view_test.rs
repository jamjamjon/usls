//! TensorView and TensorViewMut Tests
//!
//! Comprehensive tests for tensor view operations including zero-copy views,
//! mutable views, chained operations, and performance-critical patterns.

use anyhow::Result;
use usls::tensor::Tensor;

#[test]
fn test_basic_tensor_view() -> Result<()> {
    let tensor = Tensor::rand(0.0f32, 1.0f32, [6, 8, 4])?;

    // Test .view() method - creates view of entire tensor
    let full_view = tensor.view();
    assert_eq!(full_view.shape(), tensor.shape());
    assert_eq!(full_view.dtype(), usls::DType::Fp32);
    assert_eq!(full_view.ndim(), 3);
    assert_eq!(full_view.len(), 6 * 8 * 4);
    assert!(!full_view.is_empty());

    // Create immutable view using slice method
    let view = tensor.slice((1..5, 2..6, 0..4))?;

    assert_eq!(view.shape(), &[4, 4, 4]);
    assert_eq!(view.dtype(), usls::DType::Fp32);
    assert_eq!(view.ndim(), 3);
    assert_eq!(view.len(), 64);
    assert!(!view.is_empty());

    // Convert view to owned tensor
    let owned = view.to_owned()?;
    assert_eq!(owned.shape(), &[4, 4, 4]);

    Ok(())
}

#[test]
fn test_tensor_view_mut() -> Result<()> {
    let mut tensor = Tensor::zeros(vec![4, 6, 3]);

    // Test .view_mut() method - creates mutable view of entire tensor
    let mut full_view = tensor.view_mut();
    assert_eq!(full_view.shape(), &[4, 6, 3]);
    assert_eq!(full_view.dtype(), usls::DType::Fp32);
    assert_eq!(full_view.ndim(), 3);
    assert_eq!(full_view.len(), 4 * 6 * 3);

    // Test fill operation on full view
    full_view.fill(1.0)?;

    // Create mutable view using slice_mut method
    let view = tensor.slice_mut((1..3, 2..5, 0..3))?;

    assert_eq!(view.shape(), &[2, 3, 3]);
    assert_eq!(view.dtype(), usls::DType::Fp32);
    assert_eq!(view.ndim(), 3);
    assert_eq!(view.len(), 18);

    // Convert mutable view to owned tensor
    let owned = view.to_owned()?;
    assert_eq!(owned.shape(), &[2, 3, 3]);

    Ok(())
}

#[test]
fn test_view_methods() -> Result<()> {
    // Test .view() method
    let tensor = Tensor::rand(0.0f32, 1.0f32, [3, 4, 5])?;
    let view = tensor.view();

    assert_eq!(view.shape(), tensor.shape());
    assert_eq!(view.dtype(), tensor.dtype());
    assert_eq!(view.ndim(), tensor.ndim());
    assert_eq!(view.len(), tensor.len());

    // Test .view_mut() method
    let mut tensor_mut = Tensor::zeros(vec![2, 3, 4]);
    let original_shape = tensor_mut.shape().to_vec();

    {
        let mut view_mut = tensor_mut.view_mut();
        assert_eq!(view_mut.shape(), &original_shape);
        assert_eq!(view_mut.dtype(), usls::DType::Fp32);

        // Test modification through view_mut
        view_mut.fill(2.5)?;
    }

    // Verify the original tensor was modified
    let view_after = tensor_mut.view();
    if let Ok(data) = view_after.as_f32_slice() {
        assert!(data.iter().all(|&x| (x - 2.5).abs() < 1e-6));
    }

    Ok(())
}

#[test]
fn test_chained_views() -> Result<()> {
    let tensor = Tensor::rand(0.0f32, 1.0f32, [8, 8, 8, 8])?;

    // Chained view operations
    let view1 = tensor.slice((1..7, 0..8, 0..8, 0..8))?;
    assert_eq!(view1.shape(), &[6, 8, 8, 8]);

    let view2 = view1.slice((0..4, 2..6, 0..8, 0..8))?;
    assert_eq!(view2.shape(), &[4, 4, 8, 8]);

    let final_view = view2.slice((0..2, 0..4, 1..7, 2..6))?;
    assert_eq!(final_view.shape(), &[2, 4, 6, 4]);

    Ok(())
}

#[test]
fn test_sub_views() -> Result<()> {
    let tensor = Tensor::rand(0.0f32, 1.0f32, [6, 8, 4])?;
    let view = tensor.slice((1..5, 2..6, 0..4))?;

    // Create sub-view
    let sub_view = view.slice((1..3, 1..3, 1..3))?;
    assert_eq!(sub_view.shape(), &[2, 2, 2]);

    // Create mutable sub-view
    let mut tensor_mut = Tensor::zeros(vec![4, 6, 3]);
    let mut view_mut = tensor_mut.slice_mut((1..3, 2..5, 0..3))?;
    let sub_slice = view_mut.slice_mut((0..2, 1..3, 1..2))?;
    assert_eq!(sub_slice.shape(), &[2, 2, 1]);

    Ok(())
}

#[test]
fn test_image_processing_views() -> Result<()> {
    // Simulate batch image data [batch, height, width, channels]
    let batch_images = Tensor::rand(0.0f32, 1.0f32, [8, 224, 224, 3])?;

    // Single image view
    let single_image = batch_images.slice((0..1, 0..224, 0..224, 0..3))?;
    assert_eq!(single_image.shape(), &[1, 224, 224, 3]);

    // Center crop
    let cropped = batch_images.slice((0..8, 50..174, 50..174, 0..3))?;
    assert_eq!(cropped.shape(), &[8, 124, 124, 3]);

    // Channel selection (Red channel only)
    let red_channel = batch_images.slice((0..8, 0..224, 0..224, 0..1))?;
    assert_eq!(red_channel.shape(), &[8, 224, 224, 1]);

    // Batch subset
    let batch_subset = batch_images.slice((0..4, 0..224, 0..224, 0..3))?;
    assert_eq!(batch_subset.shape(), &[4, 224, 224, 3]);

    // Corner crop
    let corner_crop = batch_images.slice((0..8, 0..112, 0..112, 0..3))?;
    assert_eq!(corner_crop.shape(), &[8, 112, 112, 3]);

    Ok(())
}

#[test]
fn test_kv_cache_views() -> Result<()> {
    // Simulate KV cache [batch, num_heads, seq_len, head_dim]
    let kv_cache = Tensor::zeros(vec![4, 12, 512, 64]);

    // Single head cache view
    let single_head = kv_cache.slice((0..4, 0..1, 0..512, 0..64))?;
    assert_eq!(single_head.shape(), &[4, 1, 512, 64]);

    // Multi-head subset
    let multi_head = kv_cache.slice((0..4, 0..4, 0..512, 0..64))?;
    assert_eq!(multi_head.shape(), &[4, 4, 512, 64]);

    // Sequence window
    let seq_window = kv_cache.slice((0..4, 0..12, 100..200, 0..64))?;
    assert_eq!(seq_window.shape(), &[4, 12, 100, 64]);

    // Single batch cache
    let single_batch = kv_cache.slice((0..1, 0..12, 0..512, 0..64))?;
    assert_eq!(single_batch.shape(), &[1, 12, 512, 64]);

    // Head dimension subset
    let head_dim_subset = kv_cache.slice((0..4, 0..12, 0..512, 0..32))?;
    assert_eq!(head_dim_subset.shape(), &[4, 12, 512, 32]);

    Ok(())
}

#[test]
fn test_sequence_processing_views() -> Result<()> {
    // Simulate sequence data [batch, seq_len, hidden_dim]
    let sequence = Tensor::rand(0.0f32, 1.0f32, [16, 512, 768])?;

    // Truncate sequence
    let truncated = sequence.slice((0..16, 0..256, 0..768))?;
    assert_eq!(truncated.shape(), &[16, 256, 768]);

    // Batch subset
    let batch_subset = sequence.slice((0..8, 0..512, 0..768))?;
    assert_eq!(batch_subset.shape(), &[8, 512, 768]);

    // Feature dimension subset
    let feature_subset = sequence.slice((0..16, 0..512, 0..384))?;
    assert_eq!(feature_subset.shape(), &[16, 512, 384]);

    // Sequence window
    let sequence_window = sequence.slice((0..16, 128..384, 0..768))?;
    assert_eq!(sequence_window.shape(), &[16, 256, 768]);

    // Single sequence
    let single_sequence = sequence.slice((0..1, 0..512, 0..768))?;
    assert_eq!(single_sequence.shape(), &[1, 512, 768]);

    Ok(())
}

#[test]
fn test_performance_patterns() -> Result<()> {
    let large_tensor = Tensor::zeros(vec![100, 100, 10]); // Smaller for testing

    // Contiguous slicing (memory efficient)
    let contiguous = large_tensor.slice((10..90, 0..100, 0..10))?;
    assert_eq!(contiguous.shape(), &[80, 100, 10]);

    // Strided access
    let strided = large_tensor.slice((0..100, 10..90, 1..9))?;
    assert_eq!(strided.shape(), &[100, 80, 8]);

    // Single slice extraction
    let single_slice = large_tensor.slice((50..51, 0..100, 0..10))?;
    assert_eq!(single_slice.shape(), &[1, 100, 10]);

    Ok(())
}

#[test]
fn test_edge_cases() -> Result<()> {
    let tensor = Tensor::rand(0.0f32, 1.0f32, [8, 8, 8, 8])?;

    // Single element views
    let single_element = tensor.slice((3..4, 4..5, 5..6, 6..7))?;
    assert_eq!(single_element.shape(), &[1, 1, 1, 1]);
    assert_eq!(single_element.len(), 1);

    // Dimension reduction through indexing
    let reduced = tensor.slice((0..8, 3..4, 0..8, 0..8))?;
    assert_eq!(reduced.shape(), &[8, 1, 8, 8]);

    // Full range slicing
    let full_range = tensor.slice((0..8, 0..8, 0..8, 0..8))?;
    assert_eq!(full_range.shape(), tensor.shape());

    Ok(())
}

#[test]
fn test_complex_indexing() -> Result<()> {
    let tensor = Tensor::rand(0.0f32, 1.0f32, [8, 8, 8, 8])?;

    // Complex indexing patterns
    let complex_view = tensor.slice((0..4, 1..7, 2..6, 3..5))?;
    assert_eq!(complex_view.shape(), &[4, 6, 4, 2]);

    // Mixed range sizes
    let mixed = tensor.slice((1..2, 0..8, 3..7, 2..8))?;
    assert_eq!(mixed.shape(), &[1, 8, 4, 6]);

    Ok(())
}

#[test]
fn test_view_properties() -> Result<()> {
    let tensor = Tensor::rand(0.0f32, 1.0f32, [6, 8, 4])?;
    let view = tensor.slice((1..5, 2..6, 0..4))?;

    // Test basic properties
    assert_eq!(view.ndim(), 3);
    assert_eq!(view.len(), 64);
    assert!(!view.is_empty());
    assert_eq!(view.dtype(), usls::DType::Fp32);

    // Test empty view
    let empty_view = tensor.slice((1..1, 2..6, 0..4))?;
    assert!(empty_view.is_empty());
    assert_eq!(empty_view.len(), 0);

    Ok(())
}

#[test]
fn test_view_conversion() -> Result<()> {
    let tensor = Tensor::rand(0.0f32, 1.0f32, [4, 6, 3])?;
    let view = tensor.slice((1..3, 2..5, 0..3))?;

    // Convert view to owned tensor
    let owned = view.to_owned()?;
    assert_eq!(owned.shape(), view.shape());
    assert_eq!(owned.dtype(), view.dtype());
    assert_eq!(owned.len(), view.len());

    // Test with mutable view
    let mut tensor_mut = Tensor::zeros(vec![4, 6, 3]);
    let view_mut = tensor_mut.slice_mut((1..3, 2..5, 0..3))?;
    let owned_mut = view_mut.to_owned()?;
    assert_eq!(owned_mut.shape(), &[2, 3, 3]);

    Ok(())
}

#[test]
fn test_batch_processing() -> Result<()> {
    let large_tensor = Tensor::zeros(vec![1000, 100, 10]);
    let batch_size = 100;

    // Test batch processing pattern
    for i in (0..1000).step_by(batch_size) {
        let end = (i + batch_size).min(1000);
        let batch = large_tensor.slice((i..end, 0..100, 0..10))?;

        assert_eq!(batch.shape()[0], end - i);
        assert_eq!(batch.shape()[1], 100);
        assert_eq!(batch.shape()[2], 10);

        if i >= 200 {
            break;
        } // Only test first few batches
    }

    Ok(())
}

#[test]
fn test_tuple_syntax_consistency() -> Result<()> {
    let tensor = Tensor::rand(0.0f32, 1.0f32, [6, 8, 4])?;

    // Test various tuple syntax patterns
    let view1 = tensor.slice((1..5, 2..6, 0..4))?;
    let view2 = tensor.slice((1..5, 2..6, ..))?;
    let view3 = tensor.slice((.., 2..6, 0..4))?;

    assert_eq!(view1.shape(), &[4, 4, 4]);
    assert_eq!(view2.shape(), &[4, 4, 4]);
    assert_eq!(view3.shape(), &[6, 4, 4]);

    Ok(())
}
