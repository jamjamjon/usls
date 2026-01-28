# Cargo Features

**usls** is highly modular. Use feature flags to include only the models and hardware support you need, keeping your binary small and compilation fast.

Features in ***italics*** are enabled by default.

### Core & Utilities
  - ***`ort-download-binaries`***: Automatically download prebuilt ONNX Runtime binaries from [pyke](https://ort.pyke.io/perf/execution-providers).
  - **`ort-load-dynamic`**: Manually link ONNX Runtime. Useful for custom builds or unsupported platforms. See [Linking Guide](https://ort.pyke.io/setup/linking#static-linking) for more details.
  - **`viewer`**: Real-time image/video visualization (similar to OpenCV `imshow`). Empowered by [minifb](https://github.com/emoon/rust_minifb).
  - **`video`**: Video I/O support for reading and writing video streams. Empowered by [video-rs](https://github.com/oddity-ai/video-rs).
  - **`hf-hub`**: Download model files from Hugging Face Hub.
  - ***`annotator`***: Annotation utilities for drawing bounding boxes, keypoints, and masks on images.

### Image Formats
Additional image format support (optional for faster compilation):
  
  - **`image-all-formats`**: Enable all additional image formats.
  - **`image-gif`**, **`image-bmp`**, **`image-ico`**, **`image-avif`**, **`image-tiff`**, **`image-dds`**, **`image-exr`**, **`image-ff`**, **`image-hdr`**, **`image-pnm`**, **`image-qoi`**, **`image-tga**: Individual image format support.

### Model Categories
  - ***`vision`***: Core vision models (Detection, Segmentation, Classification, Pose, etc.).
  - **`vlm`**: Vision-Language Models (CLIP, BLIP, Florence2, etc.).
  - **`mot`**: Multi-Object Tracking utilities.
  - **`all-models`**: Enable all model categories.

### Execution Providers
Hardware acceleration for inference. Enable the one matching your hardware:

  - **`cuda`**: NVIDIA CUDA execution provider (pure model inference acceleration).
  - **`tensorrt`**: NVIDIA TensorRT execution provider (pure model inference acceleration).
  - **`nvrtx`**: NVIDIA NvTensorRT-RTX execution provider (pure model inference acceleration).
  - **`cuda-full`**: `cuda` + `cuda-runtime-build` (Model + Image Preprocessing acceleration).
  - **`tensorrt-full`**: `tensorrt` + `cuda-runtime-build` (Model + Image Preprocessing acceleration).
  - **`nvrtx-full`**: `nvrtx` + `cuda-runtime-build` (Model + Image Preprocessing acceleration).
  - **`coreml`**: Apple Silicon (macOS/iOS).
  - **`openvino`**: Intel CPU/GPU/VPU.
  - **`onednn`**: Intel Deep Neural Network Library.
  - **`directml`**: DirectML (Windows).
  - **`webgpu`**: WebGPU (Web/Chrome).
  - **`rocm`**: AMD GPU acceleration.
  - **`cann`**: Huawei Ascend NPU.
  - **`rknpu`**: Rockchip NPU.
  - **`xnnpack`**: Mobile CPU optimization.
  - **`acl`**: Arm Compute Library.
  - **`armnn`**: Arm Neural Network SDK.
  - **`azure`**: Azure ML execution provider.
  - **`migraphx`**: AMD MIGraphX.
  - **`nnapi`**: Android Neural Networks API.
  - **`qnn`**: Qualcomm SNPE.
  - **`tvm`**: Apache TVM.
  - **`vitis`**: Xilinx Vitis AI.

### CUDA Support
NVIDIA GPU acceleration with CUDA image processing kernels (requires `cudarc`):

  - **`cuda-full`**: Uses `cuda-version-from-build-system` (auto-detects via `nvcc`).
  - **`cuda-11040`**, **`cuda-11050`**, **`cuda-11060`**, **`cuda-11070`**, **`cuda-11080`**: CUDA 11.x versions (Model + Preprocess).
  - **`cuda-12000`**, **`cuda-12010`**, **`cuda-12020`**, **`cuda-12030`**, **`cuda-12040`**, **`cuda-12050`**, **`cuda-12060`**, **`cuda-12080`**, **`cuda-12090`**: CUDA 12.x versions (Model + Preprocess).
  - **`cuda-13000`**, **`cuda-13010`**: CUDA 13.x versions (Model + Preprocess).

### TensorRT Support
NVIDIA TensorRT execution provider with CUDA runtime libraries:

  - **`tensorrt-full`**: Uses `cuda-version-from-build-system` (auto-detects via `nvcc`).
  - **`tensorrt-cuda-11040`**, **`tensorrt-cuda-11050`**, **`tensorrt-cuda-11060`**, **`tensorrt-cuda-11070`**, **`tensorrt-cuda-11080`**: TensorRT + CUDA 11.x runtime.
  - **`tensorrt-cuda-12000`**, **`tensorrt-cuda-12010`**, **`tensorrt-cuda-12020`**, **`tensorrt-cuda-12030`**, **`tensorrt-cuda-12040`**, **`tensorrt-cuda-12050`**, **`tensorrt-cuda-12060`**, **`tensorrt-cuda-12080`**, **`tensorrt-cuda-12090`**: TensorRT + CUDA 12.x runtime.
  - **`tensorrt-cuda-13000`**, **`tensorrt-cuda-13010`**: TensorRT + CUDA 13.x runtime.

  > **Note**: `tensorrt-cuda-*` features enable **TensorRT execution provider** with CUDA runtime libraries for image processing. The "cuda" in the name refers to `cudarc` dependency.

### NVRTX Support
NVIDIA NvTensorRT-RTX execution provider with CUDA runtime libraries:

  - **`nvrtx-full`**: Uses `cuda-version-from-build-system` (auto-detects via `nvcc`). 
  - **`nvrtx-cuda-11040`**, **`nvrtx-cuda-11050`**, **`nvrtx-cuda-11060`**, **`nvrtx-cuda-11070`**, **`nvrtx-cuda-11080`**: NVRTX + CUDA 11.x runtime.
  - **`nvrtx-cuda-12000`**, **`nvrtx-cuda-12010`**, **`nvrtx-cuda-12020`**, **`nvrtx-cuda-12030`**, **`nvrtx-cuda-12040`**, **`nvrtx-cuda-12050`**, **`nvrtx-cuda-12060`**, **`nvrtx-cuda-12080`**, **`nvrtx-cuda-12090`**: NVRTX + CUDA 12.x runtime.
  - **`nvrtx-cuda-13000`**, **`nvrtx-cuda-13010`**: NVRTX + CUDA 13.x runtime.

  > **Note**: `nvrtx-cuda-*` features enable **NVRTX execution provider** with CUDA runtime libraries for image processing. The "cuda" in the name refers to `cudarc` dependency.

---

## 🚀 Device Combination Guide

| Scenario | Model Device (`--device`) | Processor Device (`--processor-device`) | Required Features (`-F`) |
| :--- | :--- | :--- | :--- |
| **CPU Only** | `cpu` | `cpu` | `vision` (default) |
| **GPU Inference (Slow Preprocess)** | `cuda` | `cpu` | `cuda` |
| **GPU Inference (Fast Preprocess)** | `cuda` | `cuda` | `cuda-full` or `cuda-120xxx` |
| **TensorRT (Slow Preprocess)** | `tensorrt` | `cpu` | `tensorrt` |
| **TensorRT (Fast Preprocess)** | `tensorrt` | `cuda` | `tensorrt-full` or `tensorrt-cuda-120xxx` |

> ⚠️ In multi-GPU environments (e.g., `cuda:0`, `cuda:1`), you **MUST** ensure that both `--device` and `--processor-device` use the **SAME GPU ID**. 

---

## Common Pitfalls

```toml
# ❌ Don't mix multiple CUDA versions
features = ["cuda-12040", "cuda-11080"]

# ✅ Use one execution provider
features = ["tensorrt-full"]

# ✅ Use two execution provider: cuda EP + tensorrt EP + cuda image processing
features = ["cuda-full", "tensorrt"]
features = ["cuda", "tensorrt-full"]


```
