# Device Management

usls provides flexible device control, allowing you to run the **model** (inference) and **processor** (preprocessing) on different devices for optimal performance.

!!! tip "Key Concept"
    | Component | Description | Example Device |
    | :--- | :--- | :--- |
    | **Model** | Neural network inference | `Device::Cuda(0)`, `Device::TensorRT(0)` |
    | **Image Processor** | Image preprocessing (resize, normalize) | `Device::Cpu`, `Device::Cuda(0)` |

---

## Supported Devices

| Device | Description | Required Feature |
| :--- | :--- | :--- |
| `Device::Cpu` | Standard CPU inference | (default) |
| `Device::Cuda(id)` | NVIDIA GPU via CUDA | `cuda` or `cuda-full` |
| `Device::TensorRT(id)` | NVIDIA GPU via TensorRT | `tensorrt` or `tensorrt-full` |
| `Device::NvTensorRT(id)` | NVIDIA RTX via TensorRT-RTX | `nvrtx` or `nvrtx-full` |
| `Device::CoreML` | Apple Silicon (macOS/iOS) | `coreml` |
| `Device::OpenVINO(id)` | Intel CPU/GPU/VPU | `openvino` |
| `Device::DirectML(id)` | Windows DirectML | `directml` |

!!! info "Feature Flags"
    Add to your `Cargo.toml`:
    ```toml
    [dependencies]
    usls = { git = "...", features = ["cuda"] }
    ```

---

## Device Combinations

| Scenario | Model Device | Processor Device | Feature Flag |
| :--- | :--- | :--- | :--- |
| **CPU Only** | `Device::Cpu` | `Device::Cpu` | (none) |
| **GPU Inference** | `Device::Cuda(0)` | `Device::Cpu` | `cuda` |
| **GPU Full** | `Device::Cuda(0)` | `Device::Cuda(0)` | `cuda-full` |
| **TensorRT** | `Device::TensorRT(0)` | `Device::Cpu` | `tensorrt` |
| **TensorRT Full** | `Device::TensorRT(0)` | `Device::Cuda(0)` | `tensorrt-full` |
| **TensorRT-RTX** | `Device::NvTensorRT(0)` | `Device::Cpu` | `nvrtx` |
| **TensorRT-RTX Full** | `Device::NvTensorRT(0)` | `Device::Cuda(0)` | `nvrtx-full` |
| **Apple Silicon** | `Device::CoreML` | `Device::Cpu` | `coreml` |

!!! warning "GPU Consistency"
    In multi-GPU environments, ensure both model and processor use the **same GPU ID**:
    ```rust
    // CORRECT: Both on GPU 0
    config.with_model_device(Device::Cuda(0))
          .with_processor_device(Device::Cuda(0));

    // INCORRECT: Different GPUs cause data transfer overhead
    config.with_model_device(Device::Cuda(0))
          .with_processor_device(Device::Cuda(1));
    ```

---

## Configuration Examples

### CPU Only

!!! note "Example"
    ```rust
    let config = Config::default()
        .with_model_device(Device::Cpu)
        .with_processor_device(Device::Cpu)
        .commit()?;
    ```

### GPU Inference (CPU Preprocessing)

!!! note "Example"

    ```rust
    let config = Config::default()
        .with_model_device(Device::Cuda(0))
        .with_processor_device(Device::Cpu)
        .commit()?;
    ```

### GPU Full (GPU Preprocessing)

!!! note "Example"

    ### CUDA with CUDA Preprocessing
    ```rust
    let config = Config::default()
        .with_model_device(Device::Cuda(0))
        .with_processor_device(Device::Cuda(0))
        .commit()?;
    ```

    ### TensorRT with CUDA Preprocessing

    ```rust
    let config = Config::default()
        .with_model_device(Device::TensorRT(0))
        .with_processor_device(Device::Cuda(0))
        .commit()?;
    ```

---

## Multi-GPU Support

!!! tip "Selecting GPU"
    Use device index to select specific GPU:

    ```rust
    // Use GPU 1
    let config = Config::default()
        .with_model_device(Device::Cuda(1))
        .with_processor_device(Device::Cuda(1))
        .commit()?;

    // Use GPU 2 with TensorRT
    let config = Config::default()
        .with_model_device(Device::TensorRT(2))
        .with_processor_device(Device::Cuda(2))
        .commit()?;
    ```
