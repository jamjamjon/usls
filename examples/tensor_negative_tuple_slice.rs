//! Convenient Tuple-based Negative Indexing Examples
//!
//! This example demonstrates how negative indexing with tuple syntax
//! makes tensor slicing as convenient as positive indexing.
//!
//! Key benefits:
//! - Natural syntax: tensor.slice((1..3, -1, ..))?
//! - Mix positive and negative freely
//! - Same convenience as Python/NumPy

use anyhow::Result;
use usls::tensor::Tensor;

fn main() -> Result<()> {
    println!("🦀 USLS Tensor Negative Indexing with Tuple Syntax");
    println!("================================================\n");

    // Create a 4D tensor representing a batch of RGB images
    // Shape: [batch=2, channels=3, height=4, width=8]
    let data: Vec<f32> = (0..192).map(|i| i as f32).collect();
    let tensor = Tensor::from_shape_vec(vec![2, 3, 4, 8], data)?;

    println!("Original tensor shape: {:?}\n", tensor.shape());

    // 1. Basic negative indexing - as easy as positive!
    println!("1. Basic Negative Indexing:");

    // Get the last batch
    let last_batch = tensor.slice((-1, .., .., ..))?;
    println!(
        "   Last batch: tensor.slice((-1, .., .., ..)) -> {:?}",
        last_batch.shape()
    );

    // Get the last channel of all batches
    let last_channel = tensor.slice((.., -1, .., ..))?;
    println!(
        "   Last channel: tensor.slice((.., -1, .., ..)) -> {:?}",
        last_channel.shape()
    );

    // Get the last row of all images
    let last_row = tensor.slice((.., .., -1, ..))?;
    println!(
        "   Last row: tensor.slice((.., .., -1, ..)) -> {:?}\n",
        last_row.shape()
    );

    // 2. Negative ranges - slice from the end
    println!("2. Negative Range Slicing:");

    // Get last 2 channels
    let last_channels = tensor.slice((.., -2.., .., ..))?;
    println!(
        "   Last 2 channels: tensor.slice((.., -2.., .., ..)) -> {:?}",
        last_channels.shape()
    );

    // Get last 2 rows
    let last_rows = tensor.slice((.., .., -2.., ..))?;
    println!(
        "   Last 2 rows: tensor.slice((.., .., -2.., ..)) -> {:?}",
        last_rows.shape()
    );

    // Get last 4 columns
    let last_cols = tensor.slice((.., .., .., -4..))?;
    println!(
        "   Last 4 columns: tensor.slice((.., .., .., -4..)) -> {:?}\n",
        last_cols.shape()
    );

    // 3. Negative range with both bounds
    println!("3. Negative Range with Both Bounds:");

    // Get second-to-last channel only
    let middle_channel = tensor.slice((.., -2..-1, .., ..))?;
    println!(
        "   Second-to-last channel: tensor.slice((.., -2..-1, .., ..)) -> {:?}",
        middle_channel.shape()
    );

    // Get a window in the middle-end area
    let window = tensor.slice((.., .., -3..-1, -6..-2))?;
    println!(
        "   Window slice: tensor.slice((.., .., -3..-1, -6..-2)) -> {:?}\n",
        window.shape()
    );

    // 4. Exclude elements from the end
    println!("4. Exclude Elements from End:");

    // All but last batch
    let all_but_last_batch = tensor.slice((..-1, .., .., ..))?;
    println!(
        "   All but last batch: tensor.slice((..-1, .., .., ..)) -> {:?}",
        all_but_last_batch.shape()
    );

    // All but last 2 columns
    let all_but_last_cols = tensor.slice((.., .., .., ..-2))?;
    println!(
        "   All but last 2 cols: tensor.slice((.., .., .., ..-2)) -> {:?}\n",
        all_but_last_cols.shape()
    );

    // 5. Mix positive and negative - the real power!
    println!("5. Mix Positive and Negative Indexing:");

    // First batch, last 2 channels, middle rows, last half columns
    let complex_slice = tensor.slice((0, -2.., 1..3, -4..))?;
    println!(
        "   Complex mix: tensor.slice((0, -2.., 1..3, -4..)) -> {:?}",
        complex_slice.shape()
    );

    // Second batch, first channel, last row, first 4 columns
    let another_mix = tensor.slice((1, 0, -1, ..4))?;
    println!(
        "   Another mix: tensor.slice((1, 0, -1, ..4)) -> {:?}",
        another_mix.shape()
    );

    // All batches, exclude first and last channel, exclude edges
    let inner_content = tensor.slice((.., 1..2, 1..3, 2..6))?;
    println!(
        "   Inner content: tensor.slice((.., 1..2, 1..3, 2..6)) -> {:?}\n",
        inner_content.shape()
    );

    // 6. Real-world examples
    println!("6. Real-world Use Cases:");

    // Crop bottom-right corner (common in image processing)
    let bottom_right = tensor.slice((.., .., -2.., -4..))?;
    println!(
        "   Bottom-right crop: tensor.slice((.., .., -2.., -4..)) -> {:?}",
        bottom_right.shape()
    );

    // Get RGB channels separately (all 3 channels in this case)
    let rgb_only = tensor.slice((.., .., .., ..))?; // All channels
    println!(
        "   All channels: tensor.slice((.., .., .., ..)) -> {:?}",
        rgb_only.shape()
    );

    // Batch processing: process all but the last batch
    let training_batches = tensor.slice((..-1, .., .., ..))?;
    println!(
        "   Training batches: tensor.slice((..-1, .., .., ..)) -> {:?}\n",
        training_batches.shape()
    );

    println!("✨ Summary:");
    println!("   - Negative indexing with tuples is as natural as positive indexing");
    println!("   - Mix positive and negative indices freely");
    println!("   - Perfect for end-relative operations");
    println!("   - Same syntax convenience as Python/NumPy");
    println!("   - No need for complex SliceOrIndex constructions!");

    Ok(())
}
