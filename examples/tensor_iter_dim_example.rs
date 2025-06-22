//! Tensor Dimension Iterator Example
//!
//! This example demonstrates how to use iter_dim() and iter_mut_dim() methods
//! for efficient tensor dimension iteration in various scenarios.

use anyhow::Result;
use usls::tensor::Tensor;

fn main() -> Result<()> {
    println!("🦀 Tensor Dimension Iterator Examples");
    println!("=====================================\n");

    // Example 1: Basic 2D iteration
    basic_2d_iteration()?;

    // Example 2: 3D tensor processing
    three_d_processing()?;

    // Example 3: Batch processing simulation
    batch_processing_simulation()?;

    // Example 4: Image channel processing
    image_channel_processing()?;

    // Example 5: Mutable iteration for in-place operations
    mutable_iteration_example()?;

    // Example 6: Performance comparison
    performance_comparison()?;

    Ok(())
}

/// Example 1: Basic 2D tensor iteration along different axes
fn basic_2d_iteration() -> Result<()> {
    println!("📊 Example 1: Basic 2D Iteration");
    println!("--------------------------------");

    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![3, 4], data)?;

    println!("Original tensor shape: {:?}", tensor.shape());
    println!("Original tensor data: {:?}\n", tensor.to_vec::<f32>()?);

    // Iterate along axis 0 (rows)
    println!("Iterating along axis 0 (rows):");
    for (i, row) in tensor.iter_dim(0).enumerate() {
        println!(
            "  Row {}: shape={:?}, data={:?}",
            i,
            row.shape(),
            row.to_vec::<f32>()?
        );
    }

    // Iterate along axis 1 (columns)
    println!("\nIterating along axis 1 (columns):");
    for (i, col) in tensor.iter_dim(1).enumerate() {
        println!(
            "  Column {}: shape={:?}, data={:?}",
            i,
            col.shape(),
            col.to_vec::<f32>()?
        );
    }

    println!();
    Ok(())
}

/// Example 2: 3D tensor processing (simulating video frames)
fn three_d_processing() -> Result<()> {
    println!("🎬 Example 2: 3D Tensor Processing (Video Frames)");
    println!("------------------------------------------------");

    // Simulate video data: [frames, height, width]
    let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
    let video_tensor = Tensor::from_shape_vec(vec![5, 3, 4], data)?;

    println!(
        "Video tensor shape: {:?} (frames, height, width)",
        video_tensor.shape()
    );

    // Process each frame
    println!("\nProcessing each frame:");
    for (frame_idx, frame) in video_tensor.iter_dim(0).enumerate() {
        println!("  Frame {}: shape={:?}", frame_idx, frame.shape());

        // Calculate frame statistics
        let frame_data = frame.to_vec::<f32>()?;
        let mean = frame_data.iter().sum::<f32>() / frame_data.len() as f32;
        let max_val = frame_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        println!("    Mean: {:.2}, Max: {:.2}", mean, max_val);
    }

    // Process each spatial location across time
    println!("\nAnalyzing temporal changes at each spatial location:");
    for h in 0..video_tensor.shape()[1] {
        for w in 0..video_tensor.shape()[2] {
            let temporal_slice = video_tensor.slice((.., h, w))?;
            let temporal_tensor = temporal_slice.to_owned()?;
            println!(
                "  Position ({}, {}): temporal shape={:?}",
                h,
                w,
                temporal_tensor.shape()
            );
        }
    }

    println!();
    Ok(())
}

/// Example 3: Batch processing simulation
fn batch_processing_simulation() -> Result<()> {
    println!("🔄 Example 3: Batch Processing Simulation");
    println!("----------------------------------------");

    // Simulate batch data: [batch_size, features]
    let batch_data: Vec<f32> = (0..40).map(|i| i as f32 * 0.1).collect();
    let batch_tensor = Tensor::from_shape_vec(vec![8, 5], batch_data)?;

    println!(
        "Batch tensor shape: {:?} (batch_size, features)",
        batch_tensor.shape()
    );

    // Process each sample in the batch
    let mut processed_results = Vec::new();

    println!("\nProcessing each sample:");
    for (sample_idx, sample) in batch_tensor.iter_dim(0).enumerate() {
        let sample_data = sample.to_vec::<f32>()?;

        // Simulate some processing (e.g., normalization)
        let sum: f32 = sample_data.iter().sum();
        let normalized: Vec<f32> = sample_data.iter().map(|&x| x / sum).collect();

        println!("  Sample {}: original={:?}", sample_idx, sample_data);
        println!("            normalized={:?}", normalized);

        processed_results.push(normalized);
    }

    println!(
        "\nBatch processing completed. Processed {} samples.",
        processed_results.len()
    );
    println!();
    Ok(())
}

/// Example 4: Image channel processing
fn image_channel_processing() -> Result<()> {
    println!("🖼️  Example 4: Image Channel Processing");
    println!("--------------------------------------");

    // Simulate RGB image: [height, width, channels]
    let image_data: Vec<f32> = (0..72).map(|i| i as f32).collect();
    let image_tensor = Tensor::from_shape_vec(vec![6, 4, 3], image_data)?;

    println!(
        "Image tensor shape: {:?} (height, width, channels)",
        image_tensor.shape()
    );

    // Process each channel separately
    println!("\nProcessing each channel:");
    let channel_names = ["Red", "Green", "Blue"];

    for (channel_idx, channel) in image_tensor.iter_dim(2).enumerate() {
        let channel_data = channel.to_vec::<f32>()?;

        // Calculate channel statistics
        let mean = channel_data.iter().sum::<f32>() / channel_data.len() as f32;
        let std_dev = {
            let variance = channel_data
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / channel_data.len() as f32;
            variance.sqrt()
        };

        println!(
            "  {} Channel: shape={:?}, mean={:.2}, std={:.2}",
            channel_names[channel_idx],
            channel.shape(),
            mean,
            std_dev
        );
    }

    // Process each row across all channels
    println!("\nProcessing each row:");
    for (row_idx, row) in image_tensor.iter_dim(0).enumerate() {
        println!("  Row {}: shape={:?}", row_idx, row.shape());

        // Calculate row-wise channel means
        for (c, channel_name) in channel_names.iter().enumerate() {
            let channel_slice = row.slice((.., c))?;
            let channel_tensor = channel_slice.to_owned()?;
            // For demonstration, we'll calculate mean using tensor operations
            let channel_sum = match &channel_tensor.data {
                usls::tensor::DTypeTensor::F32(arr) => arr.sum(),
                _ => 0.0,
            };
            let row_channel_mean = channel_sum / channel_tensor.len() as f32;
            println!("    {} channel mean: {:.2}", channel_name, row_channel_mean);
        }
    }

    println!();
    Ok(())
}

/// Example 5: Mutable iteration for in-place operations
fn mutable_iteration_example() -> Result<()> {
    println!("✏️  Example 5: Mutable Iteration for In-place Operations");
    println!("-------------------------------------------------------");

    let mut tensor = Tensor::zeros(vec![3, 4]);
    println!("Initial tensor shape: {:?}", tensor.shape());
    println!("Initial tensor data: {:?}\n", tensor.to_vec::<f32>()?);

    // Use mutable iterator to fill each row with different values
    {
        let mut iter = tensor.iter_mut_dim(0);
        let mut row_idx = 0;

        while let Some(mut row_slice) = iter.next_slice() {
            println!("Filling row {} with value {}.0", row_idx, row_idx + 1);
            row_slice.fill((row_idx + 1) as f32)?;
            row_idx += 1;
        }
    }

    println!("\nAfter mutable iteration:");
    println!("Final tensor data: {:?}", tensor.to_vec::<f32>()?);

    // Verify the changes by reading back
    println!("\nVerification - reading each row:");
    for (i, row) in tensor.iter_dim(0).enumerate() {
        println!("  Row {}: {:?}", i, row.to_vec::<f32>()?);
    }

    println!();
    Ok(())
}

/// Example 6: Performance comparison and iterator traits
fn performance_comparison() -> Result<()> {
    println!("⚡ Example 6: Performance and Iterator Traits");
    println!("--------------------------------------------");

    let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![100, 10], data)?;

    println!("Large tensor shape: {:?}", tensor.shape());

    // Demonstrate iterator traits
    let iter = tensor.iter_dim(0);
    println!("\nIterator traits demonstration:");
    println!("  Total length: {}", iter.total_len());
    println!("  Is empty: {}", iter.is_empty());
    println!("  Size hint: {:?}", iter.size_hint());

    // Count elements using iterator
    let count = tensor.iter_dim(0).count();
    println!("  Counted elements: {}", count);

    // Collect first few elements
    let first_three: Vec<_> = tensor.iter_dim(0).take(3).collect();
    println!(
        "  First 3 slices shapes: {:?}",
        first_three.iter().map(|t| t.shape()).collect::<Vec<_>>()
    );

    // Use enumerate
    println!("\nUsing enumerate (first 5 slices):");
    for (i, slice) in tensor.iter_dim(0).enumerate().take(5) {
        let slice_sum: f32 = slice.to_vec::<f32>()?.iter().sum();
        println!("  Slice {}: sum = {:.1}", i, slice_sum);
    }

    // Demonstrate lazy evaluation
    println!("\nLazy evaluation - creating iterator doesn't process data:");
    let lazy_iter = tensor.iter_dim(1);
    println!(
        "  Iterator created for axis 1, total_len: {}",
        lazy_iter.total_len()
    );

    // Only when we actually iterate does processing happen
    let processed_count = lazy_iter.take(5).count();
    println!(
        "  Processed only {} slices out of {}",
        processed_count,
        tensor.shape()[1]
    );

    println!();
    Ok(())
}
