# Data Loading

The `DataLoader` provides efficient, batched data ingestion from multiple sources with automatic memory management.

!!! tip "Key Features"
    - **Multi-source** — Images, videos, webcams, URLs, directories, globs
    - **Batching** — Automatic batch collation with padding
    - **Streaming** — Background thread for non-blocking iteration
    - **Progress** — Built-in progress bars for long operations

---

## Supported Sources

| Source | Example | Description |
| :--- | :--- | :--- |
| Single image | `"image.jpg"` | Load one image file |
| Directory | `"./assets/"` | All images in directory |
| Glob pattern | `"./assets/*.jpg"` | Pattern matching |
| Video file | `"video.mp4"` | Video frames as images |
| Webcam | `"0"`, `"1"` | Camera device index |
| RTSP stream | `"rtsp://..."` | Network camera stream |
| URL | `"https://.../img.jpg"` | Remote image |
| Mixed | `vec!["a.jpg", "b.png"]` or `"a.jpg | video.mp4"` | Multiple sources |

!!! example "Source Combinations"
    ```rust
    // Multiple sources at once
    let dl = DataLoader::new(vec![
        "./images/*.jpg",
        "./videos/sample.mp4",
        "https://example.com/image.png",
        "0",  // Webcam
    ])?;

    // Or use a string, separated by `|`
    let dl = DataLoader::new("a.jpg | video.mp4")?;

    ```

---

## Usage Patterns

### 1. Direct Reading (Small Data)

!!! note "Best for"
    Small datasets that fit in memory

```rust
use usls::DataLoader;

// Single image
let image = DataLoader::new("image.jpg")?.try_read_one()?;

// Nth image from collection
let image = DataLoader::new("./images/*.jpg")?.try_read_nth(2)?;

// Range of images
let images = DataLoader::new("./assets")?.try_read_range(0..5)?;

// All images
let images = DataLoader::new("./assets/*.png")?.try_read()?;
```

### 2. Streaming (Large Data/Videos)

!!! note "Best for"
    Videos, webcams, or large datasets

```rust
let dl = DataLoader::new("video.mp4")?
    .with_batch(32)           // 32 images per batch
    .with_progress_bar(true)  // Show progress
    .stream()?;               // Start background thread

for (i, (images, paths)) in dl.into_iter().enumerate() {
    println!("Batch {}: {} images", i, images.len());
    // Process batch...
}
```

---

## Configuration

| Method | Description | Default |
| :--- | :--- | :--- |
| `with_batch(n)` | Batch size | `1` |
| `with_nfv_skip(n)` | Skip n frames for video stream / webcam | `0` |
| `with_progress_bar(b)` | Show terminal progress | `true` |

!!! example "Video with frame skipping"
    ```rust
    let dl = DataLoader::new("video.mp4")?
        .with_batch(1)
        .with_nfv_skip(2)  // Skip 2 frames
        .stream()?;
    ```

---

## Integration with Models

!!! success "Complete Pipeline"
    ```rust
    use usls::*;

    fn main() -> anyhow::Result<()> {
        // Setup model
        let config = Config::yolo()
            .with_model_device(Device::Cuda(0))
            .commit()?;
        let mut model = YOLO::new(config)?;

        // Setup dataloader
        let dl = DataLoader::new("image.jpg")?
            .with_batch(model.batch())  // Use model's batch size
            .stream()?;

        // Inference loop
        for images in dl {
            let results = model.run(&images)?;
        }

        Ok(())
    }
    ```
