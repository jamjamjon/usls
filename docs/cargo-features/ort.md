# Execution Providers

Hardware acceleration for inference. Enable the one matching your hardware.

## ONNX Runtime

| Feature | Description | Default |
|---------|-------------|:-------:|
| `ort-download-binaries` | Auto-download ONNX Runtime binaries from [pyke](https://ort.pyke.io) | ✓ |
| `ort-load-dynamic` | Manual linking for custom builds. See [Linking Guide](https://ort.pyke.io/setup/linking) | x |

## Execution Providers

| Feature | Platform | Description |
|---------|----------|-------------|
| `cuda` | NVIDIA GPU | CUDA execution provider |
| `tensorrt` | NVIDIA GPU | TensorRT execution provider |
| `nvrtx` | NVIDIA GPU | NVRTX execution provider |
| `coreml` | Apple Silicon | macOS/iOS inference |
| `openvino` | Intel | CPU/GPU/VPU acceleration |
| `directml` | Windows | DirectML acceleration |
| `rocm` | AMD GPU | ROCm acceleration |
| `onednn` | Intel | Deep Neural Network Library |
| `cann` | Huawei | Ascend NPU |
| `rknpu` | Rockchip | NPU acceleration |
| `armnn` | ARM | Neural Network SDK |
| `xnnpack` | Mobile | CPU optimization |
| `webgpu` | Web | WebGPU/Chrome |
| `nnapi` | Android | Neural Networks API |
| `qnn` | Qualcomm | SNPE acceleration |
| `tvm` | - | Apache TVM |
| `azure` | Azure | ML execution provider |
| `migraphx` | AMD | MIGraphX |
| `vitis` | Xilinx | Vitis AI |

---

## CUDA Image Processor

!!! info "Prerequisites"
    Requires [cudarc](https://github.com/coreylowman/cudarc) for CUDA kernels.

Enable GPU-accelerated image preprocessing:

| Pattern | Description | Example |
|---------|-------------|---------|
| `<ep>-full` | Auto-detect CUDA version via `nvcc` | `cuda-full`, `tensorrt-full` |
| `<ep>-cuda-<ver>` | Specific CUDA version | `cuda-12040`, `tensorrt-cuda-12040` |

- **`<ep>`**: `cuda`, `tensorrt`, or `nvrtx`
- **`<ver>`**: Specific CUDA version

### Supported CUDA Versions

| Version | Features |
|---------|----------|
| 11.x | `cuda-11040`, `cuda-11050`, `cuda-11060`, `cuda-11070`, `cuda-11080` |
| 12.x | `cuda-12000`, `cuda-12010`, `cuda-12020`, `cuda-12030`, `cuda-12040`, `cuda-12050`, `cuda-12060`, `cuda-12080`, `cuda-12090` |
| 13.x | `cuda-13000`, `cuda-13010` |

!!! note "TensorRT/NVRTX Versions"
    Replace `cuda-` with `tensorrt-cuda-` or `nvrtx-cuda-` for TensorRT/NVRTX versions.
    Example: `tensorrt-cuda-12040`, `nvrtx-cuda-12080`

### Feature & Device Combinations

| Scenario | Feature | Model Device | Processor | Speed |
|----------|---------|--------------|-----------|-------|
| CPU Only | `vision` (default) | `cpu` | `cpu` | Baseline |
| CUDA | `cuda` | `cuda` | `cpu` | Slow preprocess |
| CUDA (fast) | `cuda-full` | `cuda` | `cuda` | Fast preprocess |
| TensorRT | `tensorrt` | `tensorrt` | `cpu` | Slow preprocess |
| TensorRT (fast) | `tensorrt-full` | `tensorrt` | `cuda` | Fast preprocess |

!!! tip "TensorRT EP + CUDA EP + CUDA Image Processor"
    ```toml
    features = ["tensorrt-full", "cuda"]
    # Or
    features = ["tensorrt", "cuda-full"]
    ```

!!! warning "Device Consistency"
    Different EPs can use different devices (e.g., `tensorrt:0` + `cuda:1`).
    
    However, when using **NVIDIA EP + CUDA image processor**, they **MUST** use the **same GPU ID**:
    ```toml
    # ✅ Correct: same GPU
    --device cuda:0 --processor-device cuda:0
    
    # ❌ Wrong: different GPUs
    --device cuda:0 --processor-device cuda:1
    ```


!!! danger "Don't mix CUDA versions"
    ```toml
    # ❌ Wrong
    features = ["cuda-12040", "cuda-11080"]
    
    # ✅ Correct
    features = ["tensorrt-full"]
    ```
