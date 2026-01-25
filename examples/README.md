# USLS Examples

This directory contains practical examples for all models supported by `usls`. Each demo shows how to configure, run, and integrate state-of-the-art vision and vision-language models.

## 🗂️ Contents
- [📺 How to Run Example](#how-to-run-example)
- [🚚 How to Use](#how-to-use)
- [🔍 Advanced Configuration](#advanced-configuration)
- [🌱 Contributing](#contributing)

## 🚀 How to Run Example

Execute a model demo (e.g., `RF-DETR`) with hardware acceleration:

```bash
# Model and processor on CUDA
cargo run -F cuda-full --example object-detection rfdetr --dtype fp32 --device cuda:0 --processor-device cuda:0

#  CUDA for model, CPU for image processor
cargo run -F cuda --example object-detection rfdetr --dtype fp16 --device cuda:0 --processor-device cpu

#  TensorRT for model, CUDA for image processor
cargo run -F tensorrt-full --example object-detection rfdetr --dtype fp32 --device tensorrt:0 --processor-device cuda:0

#  TensorRT for model, CPU for image processor
cargo run -F tensorrt --example object-detection rfdetr --dtype fp32 --device tensorrt:0 --processor-device cpu
```

> **Note**: Always use `--release` (or `-r`) for production to enable compiler optimizations.

### CLI Arguments

Standard arguments shared across all examples (see specific README.md for details):

| Argument | Description | Example |
|----------|-------------|---------|
| `--source` | Input media source | `path/to/img.jpg` |
| `-p`, `--prompts` | Text prompts or labels | `-p "cat"`, `-p "person" -p "dog"` |
| `--device` | Inference device | `cpu`, `cuda:0`, `tensorrt:0` |
| `--processor-device` | Pre-processing device | `cpu`, `cuda:0` |
| `--dtype` | Model precision | `fp32`, `fp16`, `q4f16`, `q8` |
| `--batch` | Inference batch size | `1`, `4`, `8` |
> **Help**: Run `cargo run --example <name> -- --help` for model-specific options.  

> **Note**: Device support requires corresponding Cargo features. DType support depends on model. See [Model Zoo](../README.md#-model-zoo) for dtype support per model.
>



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


## 🔍 Advanced Configuration

### 🧩 Module System

USLS supports complex models composed of multiple ONNX modules, enabling flexible architectures for vision and vision-language tasks.

```rust
pub enum Module {
    // Vision models (single module)
    Model,

    // Vision-Language models
    Visual,
    Textual,

    // Encoder-Decoder architectures
    Encoder,
    Decoder,
    VisualEncoder,
    TextualEncoder,
    VisualDecoder,
    TextualDecoder,
    TextualDecoderMerged,

    // Specialized components
    SizeEncoder,
    SizeDecoder,
    CoordEncoder,
    CoordDecoder,
    VisualProjection,
    TextualProjection,

    // Custom module (for extensibility)
    Custom(String),
}
```

### 🔗 API Naming Convention

USLS follows a consistent naming pattern for configuration APIs:

```
with_<module_name>_<field_name>(<value>)
```

**Examples**:
- `with_model_device()` - Set device for Model module
- `with_visual_encoder_dtype()` - Set dtype for VisualEncoder module  
- `with_textual_processor_batch()` - Set batch for TextualProcessor module

This pattern applies to **most configuration APIs** in the project, making it easy to:
- Predict API names for different modules
- Understand the purpose of each configuration option
- Maintain consistency across the codebase


### ⚙️ Execution Provider Configuration

Configure execution providers (EPs) for optimal performance across different hardware:

```rust
// Three configuration patterns:
Config::default().with_<module_name>_<ep_name>_<ep_field>(<value>)  // Per-module EP
Config::default().with_module_<ep_name>_<ep_field>(<module_name>, <value>)  // Explicit module
Config::default().with_<ep_name>_<ep_field>_all(<value>)  // Apply to all modules
```

**Examples**:
```rust
Config::dwpose_133_t()
    .with_model_coreml_static_input_shapes(true)           // Single module
    .with_module_coreml_static_input_shapes(Module::Model, true)  // Explicit module
    .with_coreml_static_input_shapes_all(true);            // All modules
```

### 🚀 TensorRT Optimization

#### TensorRT vs TensorRT-RTX

- **TensorRT EP**: Automatically handles FP32→FP16 conversion. Use `--dtype fp32` for optimal performance.
- **TensorRT-RTX EP**: Preserves input precision. No automatic conversion.

#### TensorRT Dynamic Shapes

Dynamic shapes in `usls` are configured in a way that closely mirrors `trtexec`.

**`trtexec` example:**

```bash
trtexec --fp16 --onnx=your_model.onnx \
    --minShapes=images:1x3x416x416 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:8x3x800x800 \
    --saveEngine=your_model.engine
```

**Equivalent `usls` configuration:**

```rust
Config::yolo()
    .with_model_ixx(0, 0, (1, 1, 8))        // batch: min=1, opt=1, max=8
    .with_model_ixx(0, 1, 3)                // channels: fixed at 3
    .with_model_ixx(0, 2, (416, 640, 800))  // height: min/opt/max
    .with_model_ixx(0, 3, (416, 640, 800))  // width: min/opt/max
    .commit()?;
```

**API explanation:**

* `with_<module>_ixx(input_idx, axis, (min, opt, max))` – configure dynamic shapes
* `input_idx`: input tensor index (0-based)
* `axis`: tensor dimension

  * `0` = batch
  * `1` = channel
  * `2` = height
  * `3` = width
* `(min, opt, max)`: minimum / optimal / maximum values


📘 **Source**: [`src/ort/iiix.rs`](../src/ort/iiix.rs)

### 🔧 Flexible Device & Dtype Configuration

You can configure **device** and **dtype** per module to better balance performance and memory usage.

This is particularly useful when GPU memory is limited — for example, you can place memory-intensive modules on CPU while keeping performance-critical modules on GPU.


```rust
Config::default()
    // Per-module configuration
    .with_model_device(Device::Cuda(0))
    .with_model_dtype(DType::Fp16)
    .with_visual_encoder_device(Device::Cpu)
    .with_visual_encoder_dtype(DType::Q4F16)
    .with_textual_encoder_device(Device::Cpu)
    .with_textual_encoder_dtype(DType::Fp32)
    
    // Or apply to all modules
    .with_device_all(Device::Cuda(0))
    .with_dtype_all(DType::Fp16)
    .commit()?;
```


### 📚 Additional Resources

- [Model Zoo](../README.md#-model-zoo) - Complete list of supported models
- [Configuration API](../src/config/mod.rs) - Full configuration options
- [Performance Guide](../docs/performance.md) - Optimization tips and benchmarks
- [Hardware Support](../README.md#-hardware-support) - Compatible devices and EPs


---

## 🌱 Contributing

We welcome contributions to expand the model library or improve examples.

1. **New Models**: Implement model logic in `src/models/` and provide a corresponding demo.
2. **Main Registry**: Register new examples in `Cargo.toml` with the necessary features.
3. **Consistency**: Ensure examples follow standard CLI arguments for a unified experience.

