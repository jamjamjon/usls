# USLS Examples

This directory contains practical examples for all models supported by `usls`. Each demo shows how to configure, run, and integrate state-of-the-art vision and vision-language models.

## 🚀 How to Run Example

Execute a model demo (e.g., `RF-DETR`) with hardware acceleration:

```bash
# Model + processor on CUDA
cargo run -F cuda-full --example object-detection rfdetr --device cuda:0

# CUDA model + CPU processor
cargo run -F cuda --example object-detection rfdetr --device cuda:0 --processor-device cpu

# TensorRT + CUDA processor
cargo run -F tensorrt-full --example object-detection rfdetr --device tensorrt:0

# TensorRT + CPU processor
cargo run -F tensorrt --example object-detection rfdetr --device tensorrt:0 --processor-device cpu
```

> Always use `--release` (or `-r`) for production.

---

## CLI Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--source` | Input media source | `path/to/img.jpg` |
| `-p`, `--prompts` | Text prompts | `-p "cat"` |
| `--device` | Inference device | `cpu`, `cuda:0`, `tensorrt:0` |
| `--processor-device` | Pre-processing device | `cpu`, `cuda:0` |
| `--dtype` | Model precision | `fp32`, `fp16`, `q8` |
| `--batch` | Batch size | `1`, `4` |

Run `cargo run --example <name> -- --help` for model-specific options.
