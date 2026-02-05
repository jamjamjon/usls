# Getting Started

Welcome to **usls**! 

This guide will help you get up and running with the library in just a few minutes.

## 🚀 Start with YOLO Demo
Let's run the **YOLO-Series demo** to explore models with different tasks, precision and execution providers:

- **Tasks**: `detect`, `segment`, `pose`, `classify`, `obb`
- **Versions**: `v5`, `v6`, `v7`, `v8`, `v9`, `v10`, `11`, `12`, `v13`, `26`
- **Scales**: `n`, `s`, `m`, `l`, `x`
- **Precision**: `fp32`, `fp16`, `q8`, `int8`, `q4`, `q4f16`, `bnb4`, and more
- **Execution Providers**: `CPU`, `CUDA`, `TensorRT`, `TensorRT-RTX`, `CoreML`, `OpenVINO`, and more

!!! example "First, clone the repository and navigate to the project root"

    ```bash
    git clone https://github.com/jamjamjon/usls.git
    cd usls
    ```


!!! example "Then, run the demo:"

    === "CPU"

        ```bash
        # Object detection with YOLO26n (FP16)
        cargo run -r --example yolo -- --task detect --ver 26 --scale n --dtype fp16
        ```

    === "NVIDIA (CUDA)"

        ```bash
        # Requires "cuda-full" feature
        cargo run -r -F cuda-full --example yolo -- --task segment --ver 11 --scale m --device cuda:0 --processor-device cuda:0
        ```

    === "NVIDIA (TensorRT)"

        ```bash
        # Requires "tensorrt-full" feature
        cargo run -r -F tensorrt-full --example yolo -- --device tensorrt:0 --processor-device cuda:0
        ```

    === "Apple Silicon (CoreML)"

        ```bash
        # Requires "coreml" feature
        cargo run -r -F coreml --example yolo -- --device coreml
        ```

!!! tip "For a full list of options, run:"

    ```bash
    cargo run -r --example yolo -- --help
    ```

---

## 📊 Performance Reference
!!! info ""
    **Environment:** NVIDIA RTX 3060Ti (CUDA 12.8) / Intel i5-12400F  
    **Setup:** YOLO26n, 640x640 resolution, COCO2017 val set (5,000 images)

| EP | Image<br>Processor | DType | Batch | Preprocess | Inference | Postprocess | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TensorRT | CUDA | FP16 | 1 | ~233µs | ~1.3ms | ~14µs | **~1.55ms** |
| TensorRT-RTX | CUDA | FP32 | 1 | ~233µs | ~2.0ms | ~10µs | **~2.24ms** |
| TensorRT-RTX | CUDA | FP16 | 1 | ❓ | ❓ | ❓ | ❓ |
| CUDA | CUDA | FP32 | 1 | ~233µs | ~5.0ms | ~17µs | **~5.25ms** |
| CUDA | CUDA | FP16 | 1 | ~233µs | ~3.6ms | ~17µs | **~3.85ms** |
| CUDA | CPU | FP32 | 1 | ~800µs | ~6.5ms | ~14µs | **~7.31ms** |
| CUDA | CPU | FP16 | 1 | ~800µs | ~5.0ms | ~14µs | **~5.81ms** |
| CPU | CPU | FP32 | 1 | ~970µs | ~20.5ms | ~14µs | **~21.48ms** |
| CPU | CPU | FP16 | 1 | ~970µs | ~25.0ms | ~14µs | **~25.98ms** |
| TensorRT | CUDA | FP16 | **8** | ~1.2ms | ~6.0ms | ~55µs | **~7.26ms** |
| TensorRT | CPU | FP16 | **8** | ~18.0ms | ~25.5ms | ~55µs | **~43.56ms** |

!!! tip "Multi-Batch Performance"
    When using a larger batch size (e.g., batch 8), CUDA Image processor significantly improves throughput on GPUs.

---

## Next Steps

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install usls in your own project

    [Install →](installation.md)

-   :material-code-braces:{ .lg .middle } **Integration**

    ---

    Learn how to integrate usls into your code

    [Integrate →](integration.md)

</div>
