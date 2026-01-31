# Execution Providers

Execution Providers (EPs) enable hardware-accelerated inference. All EPs support the configuration pattern: `with_<module>_<ep>_<option>()` / `with_<ep>_<option>_all()`.

!!! tip "Quick Reference"
    | Provider | Feature Flag | Device | Best For |
    | :--- | :--- | :--- | :--- |
    | **TensorRT** | `tensorrt` | `Device::TensorRT(id)` | NVIDIA GPUs (fastest) |
    | **TensorRT-RTX** | `nvrtx` | `Device::NvTensorRT(id)` | RTX GPUs |
    | **CUDA** | `cuda` | `Device::Cuda(id)` | NVIDIA GPUs |
    | **CoreML** | `coreml` | `Device::CoreML` | Apple Silicon |
    | **OpenVINO** | `openvino` | `Device::OpenVINO(target)` | Intel CPUs/GPUs |
    | **DirectML** | `directml` | `Device::DirectML(id)` | Windows |
    | **MIGraphX** | `migraphx` | `Device::MIGraphX` | AMD GPUs |
    | **CANN** | `cann` | `Device::CANN(id)` | Huawei Ascend |
    | **oneDNN** | `onednn` | `Device::OneDNN` | Intel CPUs |
    | **NNAPI** | `nnapi` | `Device::NNAPI` | Android |
    | **ARM NN** | `armnn` | `Device::ArmNN` | ARM devices |
    | **WebGPU** | `webgpu` | `Device::WebGPU` | Browsers |

---

## TensorRT

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `fp16` | `bool` | `true` | Enable FP16 precision |
| `engine_cache` | `bool` | `true` | Cache compiled engines |
| `timing_cache` | `bool` | `false` | Cache timing profiles |
| `builder_optimization_level` | `u8` | `3` | Builder optimization (0-5) |
| `max_workspace_size` | `usize` | `1073741824` | Max workspace (1GB) |
| `min_subgraph_size` | `usize` | `1` | Min subgraph node count |
| `dump_ep_context_model` | `bool` | `false` | Dump context model |
| `dump_subgraphs` | `bool` | `false` | Dump subgraphs |

!!! example "Example"
    ```rust
    Config::default()
        .with_model_device(Device::TensorRT(0))
        .with_model_tensorrt_fp16(true)
        .with_model_tensorrt_engine_cache(true)
        .with_model_tensorrt_builder_optimization_level(3)
        .commit()?;
    ```

!!! warning "First Run Slow"
    TensorRT builds engines on first run. Enable `engine_cache` for instant subsequent loads.


### Dynamic Shapes

Dynamic shapes in `usls` are configured in a way that closely mirrors `trtexec`.

**`trtexec` example:**

!!! example "Example"
    ```bash
    trtexec --fp16 --onnx=your_model.onnx \
        --minShapes=images:1x3x416x416 \
        --optShapes=images:1x3x640x640 \
        --maxShapes=images:8x3x800x800 \
        --saveEngine=your_model.engine
    ```

**Equivalent `usls` configuration:**

!!! example "Example"
    ```rust
    Config::default()
        .with_model_ixx(0, 0, (1, 1, 8))        // batch: min=1, opt=1, max=8
        .with_model_ixx(0, 1, 3)                // channels: fixed at 3
        .with_model_ixx(0, 2, (416, 640, 800))  // height: min/opt/max
        .with_model_ixx(0, 3, (416, 640, 800))  // width: min/opt/max
        .commit()?;
    ```

---

## TensorRT-RTX

Same options as TensorRT, but preserves input precision (no auto FP32→FP16 conversion).

!!! example "Example"
    ```rust
    Config::default()
        .with_model_device(Device::NvTensorRT(0))
        .commit()?;
    ```


!!! warning "TensorRT vs TensorRT-RTX"
    - **TensorRT EP**: Automatically handles FP32→FP16 conversion. Use `--dtype fp32` for optimal performance.
    - **TensorRT-RTX EP**: Preserves input precision. No automatic conversion.

---

## CUDA

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cuda_graph` | `bool` | `false` | Enable CUDA graph capture |
| `fuse_conv_bias` | `bool` | `false` | Fuse conv+bias for perf |
| `conv_max_workspace` | `bool` | `true` | Max workspace for conv search |
| `tf32` | `bool` | `true` | Enable TF32 on Ampere+ |
| `prefer_nhwc` | `bool` | `true` | Prefer NHWC layout |

!!! example "Example"
    ```rust
    Config::default()
        .with_model_device(Device::Cuda(0))
        .with_model_cuda_cuda_graph(true)
        .with_model_cuda_tf32(true)
        .commit()?;
    ```

---

## CoreML (Apple)

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `static_input_shapes` | `bool` | `true` | Static shapes for optimization |
| `subgraph_running` | `bool` | `true` | Enable subgraph mode |
| `model_format` | `u8` | `0` | `0`=MLProgram, `1`=NeuralNetwork |
| `compute_units` | `u8` | `0` | `0`=All, `1`=CPUAndGPU, `2`=CPUAndNeuralEngine, `3`=CPUOnly |
| `specialization_strategy` | `u8` | `1` | `0`=Default, `1`=FastPrediction, `2`=FastCompilation |

!!! example "Example"
    ```rust
    Config::default()
        .with_model_device(Device::CoreML)
        .with_model_coreml_static_input_shapes(true)
        .with_model_coreml_compute_units(0)
        .commit()?;
    ```

---

## OpenVINO (Intel)

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dynamic_shapes` | `bool` | `true` | Enable dynamic shapes |
| `opencl_throttling` | `bool` | `true` | Enable OpenCL throttling |
| `qdq_optimizer` | `bool` | `true` | Enable QDQ optimizer |
| `num_threads` | `usize` | `8` | Number of threads |

!!! example "Example"
    ```rust
    // CPU target
    Config::default()
        .with_model_device(Device::OpenVINO("CPU".to_string()))
        .with_model_openvino_num_threads(8)
        .commit()?;

    // GPU target
    Config::default()
        .with_model_device(Device::OpenVINO("GPU".to_string()))
        .commit()?;
    ```

!!! info "Dynamic Loading"
    Some platforms require: `cargo run -F openvino -F ort-load-dynamic`

---

## oneDNN (Intel)

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `arena_allocator` | `bool` | `true` | Enable arena allocator |

!!! example "Example"
    ```rust
    Config::default()
        .with_model_device(Device::OneDNN)
        .with_model_onednn_arena_allocator(true)
        .commit()?;
    ```

---

## CANN (Huawei)

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `graph_inference` | `bool` | `true` | Enable graph inference |
| `dump_graphs` | `bool` | `false` | Dump graphs for debug |
| `dump_om_model` | `bool` | `true` | Dump OM model |

!!! example "Example"
    ```rust
    Config::default()
        .with_model_device(Device::CANN(0))
        .with_model_cann_graph_inference(true)
        .commit()?;
    ```

---

## MIGraphX (AMD)

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `fp16` | `bool` | `true` | Enable FP16 precision |
| `exhaustive_tune` | `bool` | `false` | Exhaustive tuning |

!!! example "Example"
    ```rust
    Config::default()
        .with_model_device(Device::MIGraphX)
        .with_model_migraphx_fp16(true)
        .commit()?;
    ```

---

## NNAPI (Android)

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cpu_only` | `bool` | `false` | Force CPU-only execution |
| `disable_cpu` | `bool` | `false` | Disable CPU fallback |
| `fp16` | `bool` | `true` | Enable FP16 precision |
| `nchw` | `bool` | `false` | Use NCHW layout |

!!! example "Example"
    ```rust
    Config::default()
        .with_model_device(Device::NNAPI)
        .with_model_nnapi_fp16(true)
        .commit()?;
    ```

---

## ARM NN

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `arena_allocator` | `bool` | `true` | Enable arena allocator |

!!! example "Example"
    ```rust
    Config::default()
        .with_model_device(Device::ArmNN)
        .with_model_armnn_arena_allocator(true)
        .commit()?;
    ```

---

## WebGPU

No configurable parameters currently.

!!! example "Example"
    ```rust
    Config::default()
        .with_model_device(Device::WebGPU)
        .commit()?;
    ```

---

## CPU

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `arena_allocator` | `bool` | `true` | Enable arena allocator |

!!! example "Example"
    ```rust
    Config::default()
        .with_model_device(Device::Cpu)
        .with_model_cpu_arena_allocator(true)
        .commit()?;
    ```

---

## Configuration Patterns

| Pattern | Method | Scope |
| :--- | :--- | :--- |
| Per-module | `with_model_<ep>_<option>()` | Single module |
| Global | `with_<ep>_<option>_all()` | All modules |
| Explicit | `with_<ep>_<option>_module(Module, value)` | Specific module |

!!! example "Example"
    ```rust
    Config::default()
        // TensorRT FP16 for model module only
        .with_model_tensorrt_fp16(true)
        
        // CoreML static shapes for all modules
        .with_coreml_static_input_shapes_all(true)
        
        // Explicit module specification
        .with_tensorrt_fp16_module(Module::Visual, true)
        .commit()?;
    ```
