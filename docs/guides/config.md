# Configuration and Module System

!!! info "Core Concepts"
    usls uses two core concepts:

    - **Module System** for organizing model components
    - **Config System** for configuring everything through a unified builder API.

!!! info "API Naming Convention"
    The `Config` system in **usls** follows a builder pattern with a strict naming convention, making it easy to discover and use APIs.

    1. Per-Module: 
        - `with_<module_name>_<field_name>(<value>)`
        - `with_module_<field_name>(<module_name>, <value>)`
    2. Global: 
        - `with_<field_name>_all(<value>)`



## Module System

Models can be composed of multiple ONNX components (modules), each configurable independently.

!!! tip "Why Modules?"
    - **Flexible Device Placement**: Each module can run on different devices (GPU/CPU) independently
    - **Flexible Precision**: Set FP16, FP32, or INT8 per module as needed
    - **Flexible EP Settings**: Configure execution provider options per module

### Available Modules

| Module | Purpose | Example Models |
| :--- | :--- | :--- |
| `Model` | Default single-module | YOLO, RT-DETR |
| `Visual` / `Textual` | Vision-language components | CLIP, BLIP |
| `Encoder` / `Decoder` | Generic components | Seq2seq models |
| `VisualEncoder` / `TextualEncoder` / `TextualDecoder`<br>`VisualProjection` / `TextualProjection` | VLMs | Multi-modal |
| `SizeEncoder` / `SizeDecoder`<br>`CoordEncoder` / `CoordDecoder` | Size encoding | Specialized models |

### Usage Examples

```rust
// Single-module model
let config = Config::yolo()
    .with_model_device(Device::Cuda(0))
    .commit()?;

// Multi-module model (CLIP)
let config = Config::clip()
    .with_visual_device(Device::Cuda(0))   // GPU for images
    .with_textual_device(Device::Cpu)      // CPU for text
    .commit()?;
```

---

## Config System

### Model Basics

Configure model identity and auto-naming (YOLO-specific):

| Field | Method | Description |
| :--- | :--- | :--- |
| `name` | `with_name("yolo")` | Model identifier |
| `task` | `with_task(Task::ObjectDetection)` | Detection/Seg/Pose/Obb |
| `version` | `with_version(Version::V8)` | YOLO version |
| `scale` | `with_scale(Scale::N)` | Model size (N/S/M/L/X) |

!!! note "YOLO Auto-Naming"
    For YOLO models, the ONNX filename is auto-generated: `{version}-{scale}-{task}.onnx`
    ```rust
    Config::yolo()
        .with_version(Version::V8)
        .with_scale(Scale::N)
        .with_task(Task::ObjectDetection)
        // Auto-generates: v8-n-det.onnx
    ```

### ONNX Runtime Config

Configure inference engine settings per module:

| Field | Method | Description |
| :--- | :--- | :--- |
| `file` | `with_model_file("path.onnx")` | ONNX model file path |
| `device` | `with_model_device(Device::Cuda(0))` | Execution provider device |
| `dtype` | `with_model_dtype(DType::Fp16)` | Precision (FP32/FP16/INT8/etc) |
| `batch` | `with_model_batch(8)` | Static batch size |
| `batch_min_opt_max` | `with_model_batch_min_opt_max(1, 4, 8)` | Dynamic batch range |
| `ixx` | `with_model_ixx(0, 2, (416, 640, 800))` | Dynamic tensor dimensions |
| `num_dry_run` | `with_model_num_dry_run(3)` | Warmup iterations |
| `graph_opt_level` | `with_model_graph_opt_level(3)` | Graph optimization (0-3) |
| `num_intra_threads` | `with_model_num_intra_threads(4)` | Intra-op parallelism |
| `num_inter_threads` | `with_model_num_inter_threads(4)` | Inter-op parallelism |

!!! example "Dynamic Shapes (TensorRT)"
    ```rust
    Config::yolo()
        .with_model_device(Device::TensorRT(0))
        .with_model_ixx(0, 0, (1, 1, 8))        // Batch: min/opt/max
        .with_model_ixx(0, 2, (416, 640, 800))  // Height
        .with_model_ixx(0, 3, (416, 640, 800))  // Width
        .commit()?;
    ```

### Image Processor Config

Configure preprocessing pipeline:

| Field | Method | Description |
| :--- | :--- | :--- |
| `device` | `with_image_processor_device(Device::Cuda(0))` | Preprocessing device |
| `mean` | `with_image_mean([0.485, 0.456, 0.406])` | Normalization mean |
| `std` | `with_image_std([0.229, 0.224, 0.225])` | Normalization std |
| `resize_mode` | `with_resize_mode_type(ResizeModeType::Stretch)` | Resize strategy |
| `resize_alg` | `with_resize_alg(ResizeAlg::Nearest)` | Resize algorithm |
| `normalize` | `with_normalize(true)` | Enable normalization |
| `padding_value` | `with_padding_value(114)` | Pad value (0-255) |
| `do_resize` | `with_do_resize(true)` | Enable resizing |
| `pad_image` | `with_pad_image(true)` | Pad for super-resolution |
| `pad_size` | `with_pad_size(64)` | Pad alignment size |

### Text Processor Config (VLM)

Configure text/tokenization settings:

| Field | Method | Description |
| :--- | :--- | :--- |
| `tokenizer_file` | `with_tokenizer_file("tokenizer.json")` | Tokenizer path |
| `tokenizer_config_file` | `with_tokenizer_config_file("config.json")` | Tokenizer config |
| `special_tokens_map_file` | `with_special_tokens_map_file("special_tokens.json")` | Special tokens |
| `config_file` | `with_config_file("config.json")` | Model config |
| `model_max_length` | `with_model_max_length(2048)` | Max token length |

### Inference Parameters

Configure model-specific inference settings:

| Category | Field | Method | Default |
| :--- | :--- | :--- | :--- |
| Detection | `class_names` | `with_class_names(vec!["person"])` | `[]` |
| | `class_confs` | `with_class_confs(&[0.25])` | `[0.25]` |
| | `iou` | `with_iou(0.45)` | `None` |
| | `apply_nms` | `with_apply_nms(true)` | `None` |
| | `topk` | `with_topk(100)` | `None` |
| | `classes_excluded` | `exclude_classes(&[0])` | `[]` |
| | `classes_retained` | `retain_classes(&[0])` | `[]` |
| | `min_width` | `with_min_width(10.0)` | `None` |
| | `min_height` | `with_min_height(10.0)` | `None` |
| Keypoint | `keypoint_names` | `with_keypoint_names(vec!["nose"])` | `[]` |
| | `keypoint_confs` | `with_keypoint_confs(&[0.35])` | `[0.35]` |
| Segmentation | `num_masks` | `with_num_masks(32)` | `None` |
| | `find_contours` | `with_find_contours(true)` | `false` |
| OCR/Text | `text_names` | `with_text_names(vec!["text"])` | `[]` |
| | `text_confs` | `with_text_confs(&[0.25])` | `[0.25]` |
| | `db_unclip_ratio` | `with_db_unclip_ratio(1.5)` | `1.5` |
| | `db_binary_thresh` | `with_db_binary_thresh(0.2)` | `0.2` |
| | `max_tokens` | `with_max_tokens(256)` | `None` |
| | `token_level_class` | `with_token_level_class(true)` | `false` |
| Common | `apply_softmax` | `with_apply_softmax(true)` | `false` |

---

## Pre-configured Models

!!! info ""
    All preset configs are in `src/models/`. Each model has a `config.rs` with defaults. 
    
    Examples are in `examples/`.


---

## Full Example

!!! example "Complete Configuration"
    ```rust
    use usls::*;

    fn main() -> anyhow::Result<()> {
        let config = Config::yolo()
            // Model basics
            .with_task(Task::ObjectDetection)
            .with_version(Version::V8)
            .with_scale(Scale::N)
            
            // ONNX Runtime config
            .with_model_device(Device::TensorRT(0))
            .with_model_dtype(DType::Fp16)
            .with_model_ixx(0, 0, (1, 4, 8))        // Batch
            .with_model_ixx(0, 2, (416, 640, 800))  // Height
            .with_model_ixx(0, 3, (416, 640, 800))  // Width
            
            // Image processor
            .with_processor_device(Device::Cuda(0))
            .with_normalize(true)
            
            // Inference params
            .with_class_confs(&[0.35])
            .with_iou(0.45)
            .commit()?;

        let mut model = YOLO::new(config)?;
        Ok(())
    }
    ```

!!! danger "Always Call .commit()"
    Configuration is validated and finalized when `.commit()?` is called.
