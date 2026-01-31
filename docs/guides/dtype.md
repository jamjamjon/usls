# Data Types (DType)

usls supports multiple precision levels to balance accuracy, speed, and memory usage.

!!! tip "Quick Reference"
    | DType | Precision | Speed | Memory | Best For |
    | :--- | :--- | :--- | :--- | :--- |
    | `Fp32` | 32-bit float | Baseline | 100% | Maximum accuracy |
    | `Fp16` | 16-bit float | 2-3x faster | 50% | Modern GPUs |
    | `Q8` | 8-bit quantized | Fast | 25% | Edge deployment |
    | `Q4F16` | 4-bit + FP16 | Fast | ~15% | VLMs, limited memory |
    | `Bnb4` | BitsAndBytes 4-bit | Fast | ~12% | Ultra-low memory |

---


## Configuration

### Global (All Modules)

!!! example "Example"
    ```rust
    let config = Config::sam3()
        .with_dtype_all(DType::Fp16)
        .commit()?;
    ```

### Per-Module

!!! example "Example"
    ```rust
    let config = Config::clip()
        // Fast visual encoding
        .with_visual_dtype(DType::Fp16)
        // Accurate text encoding
        .with_textual_dtype(DType::Fp32)
        .commit()?;
    ```

---

## TensorRT Note

!!! tip "TensorRT FP32 Behavior"
    TensorRT automatically converts FP32 to FP16 for performance. Use `--dtype fp32` with TensorRT for optimal speed.
    
    TensorRT-RTX preserves the input precision exactly.