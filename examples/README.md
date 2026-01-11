# USLS Examples


This directory contains practical examples for all models supported by `usls`. Each demo shows how to configure, run, and integrate state-of-the-art vision and vision-language models.

## 🗂️ Contents
- [📺 How to Run Example](#how-to-run-example)
- [🚚 How to Use](#how-to-use)
- [🌱 Contributing](#contributing)

## 🚀 How to Run Example

Execute a model demo (e.g., `RF-DETR`) with hardware acceleration:

```bash
# Model and processor on CUDA
cargo run -F cuda --example object-detection rfdetr --dtype q4f16 --device cuda:0 --processor-device cuda:0

#  TensorRT for model, CUDA for image processor
cargo run -F tensorrt,cuda --example object-detection rfdetr --dtype fp32 --device tensorrt:0 --processor-device cuda:0
```

> **Note**: Always use `--release` (or `-r`) for production to enable compiler optimizations.

### CLI Arguments

Standard arguments shared across all examples (see specific README.md for details):

| Argument | Description | Example |
|----------|-------------|---------|
| `--source` | Input media source | `path/to/img.jpg` |
| `--device` | Inference device | `cpu`, `cuda:0`, `tensorrt:0` |
| `--processor-device` | Pre-processing device | `cpu`, `cuda:0` |
| `--dtype` | Model precision | `fp32`, `fp16`, `q4f16`, `q8` |
| `--batch` | Inference batch size | `1`, `4`, `8` |
> **Help**: Run `cargo run --example <name> -- --help` for model-specific options.  

> **Note**: Device support requires corresponding Cargo features. DType support depends on model. See [Model Zoo](../README.md#-model-zoo) for dtype support per model.




## 🚚 How to Use `usls`

`usls` implements a clean, modular pipeline from data ingestion to results visualization.

### ⚙️ Installation
Add the following to your `Cargo.toml`:

```toml
[dependencies]
# Use GitHub version
usls = { git = "https://github.com/jamjamjon/usls", features = [ "cuda" ] }

# Alternative: Use crates.io version
usls = { version = "latest-version", features = [ "cuda" ] }
```

### 🔄 Integration Workflow

1. **Configure Model**: Select pre-configured model and build
2. **Load Data**: Setup DataLoader for images/videos
3. **Run Inference**: Run inference(iterate DataLoader or run on batch images)
4. **Extract Results**: Extract data from model output
5. **Optional**: Annotate results with `Annotator`
6. **Optional**: Visualize results with `Viewer`

### 📦 Basic Usage


```rust
use usls::*;

fn main() -> anyhow::Result<()> {
    // 1. Configure & Build Model
    let config = Config::rfdetr_nano()
        .with_model_dtype(DType::Fp16)
        .with_model_device(Device::Cuda(0))
        .with_processor_device(Device::Cuda(0))
        .commit()?;
    let mut model = RFDETR::new(config)?;
    
    // 2. Setup DataLoader
    let dl = DataLoader::new("image.jpg")?
        .with_batch(model.batch())
        .with_progress_bar(true)
        .stream()?;
    
    // 3. Run Inference
    for (xs, _) in dl {
        let ys = model.run(&xs)?;
        for (x, y) in xs.iter().zip(ys.iter()) {
            println!("Detected {} objects", y.hbbs().len());
        }
    }
    Ok(())
}
```

### 🎨 Annotate Results

```rust
use usls::*;

fn main() -> anyhow::Result<()> {
    // ... model and dataloader setup ...
    
    let annotator = Annotator::default();
    
    for (xs, _) in dl {
        let ys = model.run(&xs)?;
        for (x, y) in xs.iter().zip(ys.iter()) {
            // Annotate detection results
            let image_annotated = annotator.annotate(x, y)?;
            image_annotated.save("output.jpg")?;
        }
    }
    Ok(())
}
```

### 📺 Visualize Results

```rust
use usls::*;

fn main() -> anyhow::Result<()> {
    // ... model and dataloader setup ...
    
    let mut viewer = Viewer::default().with_window_scale(1.2);

    for images in &dl {
        // Check window status
        if viewer.is_window_exist_and_closed() {
            break;
        }

        // Display frame
        viewer.imshow(&images[0])?;
        
        // Handle key events
        if let Some(key) = viewer.wait_key(30) {
            if key == usls::Key::Escape {
                break;
            }
        }
        
        // Optional: Record video
        viewer.write_video_frame(&images[0])?;
    }
    Ok(())
}
```

### 📊 Extract Results from `Y`

All models return a unified `Y` structure:

```rust
pub struct Y {
    pub texts: Vec<Text>,           // Text outputs
    pub probs: Vec<Prob>,           // Classification probabilities
    pub keypoints: Vec<Keypoint>,   // Single keypoint sets
    pub keypointss: Vec<Vec<Keypoint>>, // Multiple keypoint sets
    pub hbbs: Vec<Hbb>,             // Horizontal bounding boxes
    pub obbs: Vec<Obb>,             // Oriented bounding boxes
    pub polygons: Vec<Polygon>,     // Polygons/contours
    pub masks: Vec<Mask>,           // Segmentation masks
    pub images: Vec<Image>,         // Processed images
    pub embedding: X,               // Feature embeddings
    pub extras: HashMap<String, X>, // Additional outputs
}
```

**Access Patterns**:

```rust
let y = model.run(&xs)?;

// Reference access (no ownership transfer)
let detections = y.hbbs();        // &[Hbb]
let texts = y.texts();            // &[Text]
let embedding = y.embedding();    // &X

// Owned access (transfers ownership)
let detections = y.hbbs;          // Vec<Hbb>
let texts = y.texts;              // Vec<Text>
let embedding = y.embedding;      // X
```

📘 **Source**: [`src/results/y.rs`](../src/results/y.rs)


---

## 🌱 Contributing

We welcome contributions to expand the model library or improve examples.

1. **New Models**: Implement model logic in `src/models/` and provide a corresponding demo.
2. **Main Registry**: Register new examples in `Cargo.toml` with the necessary features.
3. **Consistency**: Ensure examples follow standard CLI arguments for a unified experience.

