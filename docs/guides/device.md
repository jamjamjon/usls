# Device Management

**usls** allows you to precisely control where model inference and image preprocessing occur.

## Supported Devices

| Device | Description | Feature |
|--------|-------------|---------|
| `Cpu` | Standard CPU inference | (Default) |
| `Cuda(id)` | NVIDIA GPU via CUDA | `cuda` |
| `TensorRT(id)` | NVIDIA GPU via TensorRT | `tensorrt` |
| `CoreML` | Apple Silicon | `coreml` |
| `OpenVINO(id)` | Intel CPU/GPU/VPU | `openvino` |

## Model vs. Processor Device

A unique feature of **usls** is the ability to separate the device for the **Model** (Inference) and the **Processor** (Preprocessing).

```rust
let config = Config::clip()
    .with_model_device(Device::Cuda(0))     // Inference on GPU
    .with_processor_device(Device::Cpu)     // Preprocessing on CPU
    .commit()?;
```

!!! info "Performance Tip"
    For small models, CPU preprocessing is often fast enough. For large batches or high-resolution images, use `Device::Cuda(0)` for the processor to enable GPU-accelerated resizing and normalization.

## Multi-GPU Support

In multi-GPU environments, ensure both the model and the processor use the same GPU ID if they are intended to work together efficiently.

```rust
let gpu_id = 1;
config.with_model_device(Device::Cuda(gpu_id))
      .with_processor_device(Device::Cuda(gpu_id));
```