# DType & Precision

**usls** supports multiple data types (DTypes) for model inference, allowing you to balance accuracy and performance.

## Supported DTypes

| DType | Description | Best For |
|-------|-------------|----------|
| `Fp32` | 32-bit floating point | Maximum accuracy (default) |
| `Fp16` | 16-bit floating point | Best performance on modern GPUs |
| `Q8` | 8-bit quantized | Reduced memory, high speed |
| `Q4F16` | 4-bit quantized (Fp16) | VLM models on restricted memory |
| `Bnb4` | BitsAndBytes 4-bit | Ultra-low memory usage |

## Configuration

You can set the DType for the entire model or per-module:

```rust
let config = Config::clip()
    .with_dtype_all(DType::Fp16) // Global
    .with_visual_dtype(DType::Fp16) // Per-module
    .commit()?;
```

!!! tip "Hardware Support"
    Ensure your chosen Execution Provider supports the target DType. For example, `Fp16` is highly recommended for **TensorRT** and **CUDA** on modern NVIDIA GPUs.