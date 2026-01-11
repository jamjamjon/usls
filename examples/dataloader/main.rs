use usls::DataLoader;

// TODO: remove

#[path = "../utils/mod.rs"]
mod utils;

/// This example demonstrates how to use the `DataLoader` to ingest data from various sources.
///
/// `DataLoader` is the core data ingestion engine in `usls`, supporting:
/// - **Images**: Local files, remote URLs, directories, and glob patterns.
/// - **Videos**: Local files, remote streams (RTSP/RTMP/HTTP).
/// - **Hardware**: Webcams/Cameras (via device index).
/// - **Hybrid**: Any combination of the above.
fn main() -> anyhow::Result<()> {
    utils::init_logging();

    // =========================================================================
    // 1. Synchronous Loading (`try_read_*` series)
    // Best for static images. Returns `Vec<Image>` or `Image`.
    // =========================================================================

    // --- Single Item ---
    // Read the first valid image from a source
    let image = DataLoader::new("./assets/bus.jpg")?.try_read_one()?;
    println!("## DataLoader: loaded 1 image with try_read_one():");
    println!("  - {:?}", image);

    // Read the Nth valid image (0-indexed)
    let image = DataLoader::new("./assets/*.jpg")?.try_read_nth(2)?;
    println!("## DataLoader: loaded n-th image with try_read_nth():");
    println!("  - {:?}", image);

    // --- Range of Items ---
    // Read a specific range of images using Rust range syntax
    let images = DataLoader::new("./assets")?.try_read_range(0..2)?;
    println!(
        "## DataLoader: loaded {} images with try_read_range()",
        images.len()
    );
    for image in images {
        println!("  - {:?}", image);
    }

    // --- Batch Loading ---
    // Read all valid images from a collection of mixed sources.
    // Sources can be a Vec or a '|' separated string.
    let images = DataLoader::new(vec![
        "./assets/cat.png",  // Local file
        "images/bus.jpg",    // GitHub release image 
        "https://fastly.picsum.photos/id/237/200/300.jpg?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U", // Remote URL
        "./assets/",         // Directory (loads all images inside)
        "./assets/*.jpg",    // Glob pattern
    ])?
    .try_read()?;

    println!(
        "## DataLoader: loaded {} images with try_read()",
        images.len()
    );
    for image in images {
        println!("  - {:?}", image);
    }

    // =========================================================================
    // 2. Streaming (`stream()`)
    // Frames are processed in a background thread and yielded in batches.
    // =========================================================================

    // // Mixed source: image | video | glob | webcam
    // let source = "./assets/bus.jpg | ../7.mp4 | ./assets/*.png | 0";

    // let dl = DataLoader::new(source)?
    //     .with_batch(1) // Number of images per yielded batch
    //     // .with_nfv_skip(0) // Skip n frames for every 1 processed (Video/Stream only)
    //     .with_progress_bar(true) // Enable terminal progress bar
    //     .stream()?; // Starts the background producer thread

    // // Iterate over the stream. Each iteration yields a `Vec<Image>`.
    // for (i, batch) in dl.into_iter().enumerate() {
    //     println!("Batch-{}: received {} images", i, batch.len());

    //     // Example: access individual images in the batch
    //     for _img in batch {
    //         // Process image...
    //     }

    // }

    Ok(())
}
