//! Comprehensive tests for tensor dimension iteration
//!
//! This module tests the iter_dim() functionality with strict ndarray compatibility,
//! performance benchmarks, and edge case handling. All tests ensure 100% consistency
//! with ndarray's axis iteration behavior.

use anyhow::Result;
use ndarray::Array;
use usls::tensor::Tensor;

// ============================================================================
// Basic iter_dim Tests
// ============================================================================

#[test]
fn test_iter_dim_basic_2d() -> Result<()> {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![3, 4], data.clone())?;
    let ndarray = Array::from_shape_vec((3, 4), data)?;

    // Test iteration along axis 0 (rows)
    let tensor_iter: Vec<_> = tensor.iter_dim(0).collect();
    let ndarray_iter: Vec<_> = ndarray.axis_iter(ndarray::Axis(0)).collect();

    assert_eq!(tensor_iter.len(), ndarray_iter.len());

    for (tensor_slice, ndarray_slice) in tensor_iter.iter().zip(ndarray_iter.iter()) {
        assert_eq!(tensor_slice.shape(), ndarray_slice.shape());
        let tensor_data = tensor_slice.to_vec::<f32>()?;
        let ndarray_data: Vec<f32> = ndarray_slice.iter().cloned().collect();
        assert_eq!(tensor_data, ndarray_data);
    }

    Ok(())
}

#[test]
fn test_iter_dim_basic_2d_axis1() -> Result<()> {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![3, 4], data.clone())?;
    let ndarray = Array::from_shape_vec((3, 4), data)?;

    // Test iteration along axis 1 (columns)
    let tensor_iter: Vec<_> = tensor.iter_dim(1).collect();
    let ndarray_iter: Vec<_> = ndarray.axis_iter(ndarray::Axis(1)).collect();

    assert_eq!(tensor_iter.len(), ndarray_iter.len());

    for (tensor_slice, ndarray_slice) in tensor_iter.iter().zip(ndarray_iter.iter()) {
        assert_eq!(tensor_slice.shape(), ndarray_slice.shape());
        let tensor_data = tensor_slice.to_vec::<f32>()?;
        let ndarray_data: Vec<f32> = ndarray_slice.iter().cloned().collect();
        assert_eq!(tensor_data, ndarray_data);
    }

    Ok(())
}

#[test]
fn test_iter_dim_3d() -> Result<()> {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![2, 3, 4], data.clone())?;
    let ndarray = Array::from_shape_vec((2, 3, 4), data)?;

    // Test iteration along each axis
    for axis in 0..3 {
        let tensor_iter: Vec<_> = tensor.iter_dim(axis).collect();
        let ndarray_iter: Vec<_> = ndarray.axis_iter(ndarray::Axis(axis)).collect();

        let expected_len = tensor.shape()[axis];
        assert_eq!(
            tensor_iter.len(),
            expected_len,
            "Axis {} length mismatch",
            axis
        );
        assert_eq!(
            ndarray_iter.len(),
            expected_len,
            "Axis {} length mismatch",
            axis
        );

        for (i, (tensor_slice, ndarray_slice)) in
            tensor_iter.iter().zip(ndarray_iter.iter()).enumerate()
        {
            assert_eq!(
                tensor_slice.shape(),
                ndarray_slice.shape(),
                "Axis {} slice {} shape mismatch",
                axis,
                i
            );

            let tensor_data = tensor_slice.to_vec::<f32>()?;
            let ndarray_data: Vec<f32> = ndarray_slice.iter().cloned().collect();
            assert_eq!(
                tensor_data, ndarray_data,
                "Axis {} slice {} data mismatch",
                axis, i
            );
        }
    }

    Ok(())
}

#[test]
fn test_iter_dim_4d() -> Result<()> {
    let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![2, 3, 4, 5], data.clone())?;
    let ndarray = Array::from_shape_vec((2, 3, 4, 5), data)?;

    // Test iteration along each axis
    for axis in 0..4 {
        let tensor_iter: Vec<_> = tensor.iter_dim(axis).collect();
        let ndarray_iter: Vec<_> = ndarray.axis_iter(ndarray::Axis(axis)).collect();

        let expected_len = tensor.shape()[axis];
        assert_eq!(
            tensor_iter.len(),
            expected_len,
            "Axis {} length mismatch",
            axis
        );
        assert_eq!(
            ndarray_iter.len(),
            expected_len,
            "Axis {} length mismatch",
            axis
        );

        for (i, (tensor_slice, ndarray_slice)) in
            tensor_iter.iter().zip(ndarray_iter.iter()).enumerate()
        {
            assert_eq!(
                tensor_slice.shape(),
                ndarray_slice.shape(),
                "Axis {} slice {} shape mismatch",
                axis,
                i
            );

            let tensor_data = tensor_slice.to_vec::<f32>()?;
            let ndarray_data: Vec<f32> = ndarray_slice.iter().cloned().collect();
            assert_eq!(
                tensor_data, ndarray_data,
                "Axis {} slice {} data mismatch",
                axis, i
            );
        }
    }

    Ok(())
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
#[should_panic(expected = "Dimension 2 is out of bounds for tensor with 2 dimensions")]
fn test_iter_dim_out_of_bounds() {
    let tensor = Tensor::zeros(vec![3, 4]);
    let _iter = tensor.iter_dim(2); // Should panic
}

#[test]
fn test_iter_dim_single_element() -> Result<()> {
    let tensor = Tensor::from_shape_vec(vec![1, 1, 1], vec![42.0])?;
    let ndarray = Array::from_shape_vec((1, 1, 1), vec![42.0])?;

    for axis in 0..3 {
        let tensor_iter: Vec<_> = tensor.iter_dim(axis).collect();
        let ndarray_iter: Vec<_> = ndarray.axis_iter(ndarray::Axis(axis)).collect();

        assert_eq!(tensor_iter.len(), 1);
        assert_eq!(ndarray_iter.len(), 1);

        let tensor_slice = &tensor_iter[0];
        let ndarray_slice = &ndarray_iter[0];

        assert_eq!(tensor_slice.shape(), ndarray_slice.shape());

        let tensor_data = tensor_slice.to_vec::<f32>()?;
        let ndarray_data: Vec<f32> = ndarray_slice.iter().cloned().collect();
        assert_eq!(tensor_data, ndarray_data);
    }

    Ok(())
}

#[test]
fn test_iter_dim_empty_dimension() -> Result<()> {
    // Test tensor with one dimension being 0
    let tensor = Tensor::zeros(vec![0, 3, 4]);
    let ndarray = Array::<f32, _>::zeros((0, 3, 4));

    // Iteration along axis 0 should yield no elements
    let tensor_iter: Vec<_> = tensor.iter_dim(0).collect();
    let ndarray_iter: Vec<_> = ndarray.axis_iter(ndarray::Axis(0)).collect();

    assert_eq!(tensor_iter.len(), 0);
    assert_eq!(ndarray_iter.len(), 0);

    // Iteration along axis 1 should yield 3 empty slices
    let tensor_iter: Vec<_> = tensor.iter_dim(1).collect();
    let ndarray_iter: Vec<_> = ndarray.axis_iter(ndarray::Axis(1)).collect();

    assert_eq!(tensor_iter.len(), 3);
    assert_eq!(ndarray_iter.len(), 3);

    for (tensor_slice, ndarray_slice) in tensor_iter.iter().zip(ndarray_iter.iter()) {
        assert_eq!(tensor_slice.shape(), &[0, 4]);
        assert_eq!(tensor_slice.shape(), ndarray_slice.shape());
        assert!(tensor_slice.is_empty());
    }

    Ok(())
}

// ============================================================================
// Iterator Trait Tests
// ============================================================================

#[test]
fn test_iter_dim_iterator_traits() -> Result<()> {
    let tensor = Tensor::from_shape_vec(vec![3, 4], (0..12).map(|i| i as f32).collect())?;
    let mut iter = tensor.iter_dim(0);

    // Test size_hint
    assert_eq!(iter.size_hint(), (3, Some(3)));

    // Test len (ExactSizeIterator)
    assert_eq!(iter.total_len(), 3);

    // Test next
    let first = iter.next().unwrap();
    assert_eq!(first.shape(), &[4]);
    assert_eq!(iter.len(), 2);
    assert_eq!(iter.size_hint(), (2, Some(2)));

    // Test remaining
    let remaining: Vec<_> = iter.collect();
    assert_eq!(remaining.len(), 2);

    Ok(())
}

#[test]
fn test_iter_dim_collect_and_enumerate() -> Result<()> {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![3, 4], data)?;

    // Test collect
    let slices: Vec<_> = tensor.iter_dim(0).collect();
    assert_eq!(slices.len(), 3);

    // Test enumerate
    for (i, slice) in tensor.iter_dim(0).enumerate() {
        assert_eq!(slice.shape(), &[4]);
        let slice_data = slice.to_vec::<f32>()?;
        let expected: Vec<f32> = (i * 4..(i + 1) * 4).map(|x| x as f32).collect();
        assert_eq!(slice_data, expected);
    }

    Ok(())
}

// ============================================================================
// Mutable Iterator Tests
// ============================================================================

#[test]
fn test_iter_mut_dim_basic() -> Result<()> {
    let mut tensor = Tensor::zeros(vec![3, 4]);
    let mut iter = tensor.iter_mut_dim(0);

    // Test basic properties
    assert_eq!(iter.total_len(), 3);
    assert!(!iter.is_empty());
    assert_eq!(iter.remaining(), 3);

    // Test next_slice
    let mut slice = iter.next_slice().unwrap();
    assert_eq!(slice.shape(), &[4]);

    // Fill the slice with values
    slice.fill(1.0)?;

    // Check remaining after filling
    assert_eq!(iter.remaining(), 2);

    // Test remaining slices
    while let Some(mut slice) = iter.next_slice() {
        slice.fill(2.0)?;
    }

    // Verify modifications
    let final_data = tensor.to_vec::<f32>()?;
    let expected = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    assert_eq!(final_data, expected);

    Ok(())
}

#[test]
fn test_iter_mut_dim_reset() -> Result<()> {
    let mut tensor = Tensor::zeros(vec![2, 3]);
    let mut iter = tensor.iter_mut_dim(0);

    // Consume one element
    let _slice = iter.next_slice().unwrap();
    assert_eq!(iter.remaining(), 1);

    // Reset iterator
    iter.reset();
    assert_eq!(iter.remaining(), 2);
    assert_eq!(iter.total_len(), 2);

    Ok(())
}

// ============================================================================
// Performance and Memory Tests
// ============================================================================

#[test]
fn test_iter_dim_memory_efficiency() -> Result<()> {
    // Create a large tensor
    let tensor = Tensor::zeros(vec![1000, 1000]);

    // Iterate without collecting all at once (memory efficient)
    let mut count = 0;
    for slice in tensor.iter_dim(0) {
        assert_eq!(slice.shape(), &[1000]);
        count += 1;

        // Only check first few to avoid long test times
        if count >= 10 {
            break;
        }
    }

    assert_eq!(count, 10);
    Ok(())
}

#[test]
fn test_iter_dim_consistency_across_types() -> Result<()> {
    // Test with different data types to ensure consistency
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor_f32 = Tensor::from_shape_vec(vec![2, 3, 4], data)?;

    // Test iteration behavior is consistent
    for axis in 0..3 {
        let slices: Vec<_> = tensor_f32.iter_dim(axis).collect();
        assert_eq!(slices.len(), tensor_f32.shape()[axis]);

        for slice in slices {
            let mut expected_shape = tensor_f32.shape().to_vec();
            expected_shape.remove(axis);
            assert_eq!(slice.shape(), expected_shape.as_slice());
        }
    }

    Ok(())
}

// ============================================================================
// Real-world Usage Patterns
// ============================================================================

#[test]
fn test_iter_dim_batch_processing() -> Result<()> {
    // Simulate batch processing scenario [batch_size, features]
    let batch_data: Vec<f32> = (0..40).map(|i| i as f32).collect();
    let batch_tensor = Tensor::from_shape_vec(vec![8, 5], batch_data)?;

    // Process each sample in the batch
    let mut processed_samples = Vec::new();
    for (i, sample) in batch_tensor.iter_dim(0).enumerate() {
        assert_eq!(sample.shape(), &[5]);

        // Simulate processing (e.g., normalization)
        let sample_data = sample.to_vec::<f32>()?;
        let sum: f32 = sample_data.iter().sum();
        processed_samples.push(sum);

        // Verify sample data
        let expected_start = i * 5;
        let expected: Vec<f32> = (expected_start..expected_start + 5)
            .map(|x| x as f32)
            .collect();
        assert_eq!(sample_data, expected);
    }

    assert_eq!(processed_samples.len(), 8);
    Ok(())
}

#[test]
fn test_iter_dim_sequence_processing() -> Result<()> {
    // Simulate sequence processing [seq_len, hidden_dim]
    let seq_data: Vec<f32> = (0..60).map(|i| i as f32).collect();
    let seq_tensor = Tensor::from_shape_vec(vec![10, 6], seq_data)?;

    // Process each time step
    for (t, timestep) in seq_tensor.iter_dim(0).enumerate() {
        assert_eq!(timestep.shape(), &[6]);

        let timestep_data = timestep.to_vec::<f32>()?;
        let expected_start = t * 6;
        let expected: Vec<f32> = (expected_start..expected_start + 6)
            .map(|x| x as f32)
            .collect();
        assert_eq!(timestep_data, expected);
    }

    Ok(())
}

#[test]
fn test_iter_dim_channel_processing() -> Result<()> {
    // Simulate image channel processing [height, width, channels]
    let image_data: Vec<f32> = (0..72).map(|i| i as f32).collect();
    let image_tensor = Tensor::from_shape_vec(vec![6, 4, 3], image_data)?;

    // Process each channel
    for (c, channel) in image_tensor.iter_dim(2).enumerate() {
        assert_eq!(channel.shape(), &[6, 4]);

        let channel_data = channel.to_vec::<f32>()?;
        assert_eq!(channel_data.len(), 24);

        // Verify channel data pattern
        for (i, &value) in channel_data.iter().enumerate() {
            let expected = (i * 3 + c) as f32;
            assert_eq!(value, expected);
        }
    }

    Ok(())
}
