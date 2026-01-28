# Data Loading

**usls** provides a flexible and efficient `DataLoader` system designed for high-performance inference. It supports various input sources, automatic batching, and parallel processing.

## 📥 Supported Sources

The `DataLoader` can ingest data from multiple formats:

- **Images**: Single files (`.jpg`, `.png`), directories, or lists of paths.
- **Videos**: Local video files or RTSP/HTTP streams.
- **Hardware**: Webcams and other camera devices.
- **Memory**: Raw bytes or pre-loaded images.

## 🔄 Basic Usage

### Read into Image or Vec<Image>
```rust
use usls::DataLoader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
// Single image
let image = DataLoader::new("image.jpg")?.try_read_one()?;

// Nth image from collection
let image = DataLoader::new("./images/*.jpg")?.try_read_nth(2)?;

 // Range of images
 let images = DataLoader::new("./assets")?.try_read_range(0..5)?;

 // All images from mixed sources
 let images = DataLoader::new(vec![
     "local.jpg",
     "https://example.com/remote.jpg",
     "./images/*.png",
 ])?.try_read()?;
 Ok(())
}
```

### Video & Webcam
```rust
use usls::DataLoader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let source = "./assets/bus.jpg | ../video.mp4 | ./assets/*.png | 0";
let dl = DataLoader::new(source)?
    .with_batch(32)           // 32 images per batch
    .with_progress_bar(true)  // Show progress
    .stream()?;               // Start background thread

for (i, batch) in dl.into_iter().enumerate() {
    println!("Batch {}: {} images", i, batch.len());
    // Process batch...
}
Ok(())
}
```

## 🎛️ Configuration

| Method | Description | Default |
| :--- | :--- | :--- |
| `with_batch(n)` | Set the batch size | `1` |
| `with_fps(n)` | Limit processing speed for videos/streams | `None` |
| `with_progress_bar(bool)` | Show a progress bar in the terminal | `true` |

***TODO***