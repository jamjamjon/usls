# FAQ

***TODO***

<!-- 

Frequently asked questions about **usls**.
## 🚀 General

??? question "How do I install usls?"
    Add usls to your `Cargo.toml`. See the [Installation](getting-started/installation.md) guide for details.

??? question "What are the minimum requirements?"
    - Rust 1.87+
    - ONNX Runtime 1.22.0+
    - (Optional) CUDA 11/12 or TensorRT 10 for GPU acceleration.

## 🔧 Configuration

??? question "CUDA vs. TensorRT?"
    - **CUDA**: Easier setup, good flexibility.
    - **TensorRT**: Maximum performance, requires an initial engine building step.

??? question "Model Device vs. Processor Device?"
    - **Model Device**: Where the neural network runs (Inference).
    - **Processor Device**: Where image resizing and normalization happen (Preprocessing).

## 🎯 Model Usage

??? question "Can I use custom models?"
    Yes, you can load any ONNX model using `Config::from_file("model.onnx")`.

??? question "How do I optimize for speed?"
    - Use `DType::Fp16`.
    - Enable `TensorRT`.
    - Use GPU preprocessing (`Device::Cuda(0)` for both model and processor).

## 🐛 Troubleshooting

??? question "CUDA not available error?"
    Ensure you have the NVIDIA drivers and CUDA toolkit installed, and that you've enabled the `cuda` or `cuda-full` feature in your `Cargo.toml`.

??? question "Out of Memory (OOM)?"
    - Reduce the batch size.
    - Use half-precision (`Fp16`).
    - Use quantized models (`Q4`, `Q8`).

--- -->

!!! info "Still have questions?"
    If you can't find what you're looking for, feel free to open a [GitHub Issue](https://github.com/jamjamjon/usls/issues).
