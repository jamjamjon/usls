# Execution Providers

**usls** supports multiple Execution Providers (EPs) via ONNX Runtime to optimize inference on different hardware platforms.

## Configuration Patterns

You can configure EPs at different levels of granularity using a predictable naming convention:

1.  **Per-Module**: `with_<module>_<ep>_<field>(value)`
2.  **Global (All Modules)**: `with_<ep>_<field>_all(value)`

### Example: CoreML Configuration

```rust
Config::dwpose()
    .with_model_coreml_static_input_shapes(true) // Just for the 'model' module
    .with_coreml_static_input_shapes_all(true)    // For all modules in this config
    .commit()?;
```

## TensorRT Optimization

TensorRT provides the best performance on NVIDIA GPUs but requires an initial "engine building" phase.

### Dynamic Shapes

**usls** uses a syntax that closely mirrors `trtexec`. You define the minimum, optimal, and maximum shapes for each axis.

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
    .with_model_ixx(0, 0, (1, 1, 8))        // Axis 0 (Batch): min=1, opt=1, max=8
    .with_model_ixx(0, 1, 3)                // Axis 1 (Channels): fixed at 3
    .with_model_ixx(0, 2, (416, 640, 800))  // Axis 2 (Height): min=416, opt=640, max=800
    .with_model_ixx(0, 3, (416, 640, 800))  // Axis 3 (Width): min=416, opt=640, max=800
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

### TensorRT vs. TensorRT-RTX

- **TensorRT**: Automatically handles FP32 to FP16 conversion.
- **TensorRT-RTX**: Preserves input precision and uses specific RTX optimizations.

## Supported Providers

| Provider | Device | Feature |
| :--- | :--- | :--- |
| **CUDA** | `Device::Cuda(id)` | `cuda` |
| **TensorRT** | `Device::TensorRT(id)` | `tensorrt` |
| **CoreML** | `Device::CoreML` | `coreml` |
| **OpenVINO** | `Device::OpenVINO(id)` | `openvino` |
| **DirectML** | `Device::DirectML(id)` | `directml` |

---

!!! info "Engine Caching"
    For TensorRT, usls automatically handles engine caching. The first run will be slow while the engine builds, but subsequent runs will start instantly.
