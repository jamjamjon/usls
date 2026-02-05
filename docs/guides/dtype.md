# Data Types (DType) and Quantization

usls provides multiple precision ONNX models for each algorithm, using `DType` for model selection and EP configuration.

!!! tip "Quick Reference"
    | DType | Precision | Description |
    | :--- | :--- | :--- |
    | `fp32` | 32-bit float | Original precision, exported from PyTorch/TensorFlow |
    | `fp16` | 16-bit float | Mixed precision fp16/fp32  |
    | `q8` | 8-bit | Dynamic quantization using `onnxruntime.quantization.quantize_dynamic`.<br><br>Auto-selects `int8/uint8` based on operators (`Conv`, `GroupQueryAttention`, `MultiHeadAttention` → `uint8`, otherwise `int8`) |
    | `int8` | 8-bit  | Static quantization for TensorRT EP using NVIDIA Model-Optimizer.<br><br>Note: May fail on non-TensorRT EPs |
    | `s8s8` | 8-bit  | Static quantization (signed weights, signed activations), using `onnxruntime.quantization.quantize_static` |
    | `s8u8` | 8-bit  | Static quantization (signed weights, unsigned activations) |
    | `u8u8` | 8-bit  | Static quantization (unsigned weights, unsigned activations) |
    | `q4` | 4-bit | Float32 to 4-bit int using `onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer` |
    | `q4f16` | 4-bit + 16-bit float | `q4` with cast to fp16 |
    | `bnb4` | 4-bit | 4-bit quantization using `FP4/NF4` via `onnxruntime.quantization.matmul_bnb4_quantizer.MatMulBnb4Quantizer` | 


---

## Basic Usage

!!! example "Global (All Modules)"
    ```rust
    let config = Config::sam3()
        .with_dtype_all(DType::Fp16)
        .commit()?;
    ```

!!! example "Per-Module"
    ```rust
    let config = Config::clip()
        // Fast visual encoding
        .with_visual_dtype(DType::Fp16)
        // Accurate text encoding
        .with_textual_dtype(DType::Fp32)
        .commit()?;
    ```

!!! danger "Model Availability"
    usls does not provide all quantization variants for every model. Refer to the quantization schemes in this guide to create your own quantized models as needed.

---

## TensorRT EP

!!! tip "Use FP16"
    TensorRT automatically converts FP32 to FP16 for performance. Use FP32 models with TensorRT for optimal speed:
    
    ```rust
    Config::default().with_model_file("<fp32-model>.onnx")
    ```
    
    Or use `--dtype fp32` in usls examples.

!!! tip "Use int8/FP8/Int4"
    #### Method 1: NVIDIA Model-Optimizer
    - Step 1: Quantize FP32 ONNX to int8/FP8/Int4 QDQ ONNX using [NVIDIA Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)
    - Step 2: Feed quantized QDQ ONNX directly to TensorRT EP
    - [Demo: Model-Optimizer onnx_ptq](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq)

    #### Method 2: ONNX Runtime Quantization
    - Step 1: Provide FP32 ONNX model and calibration table
    - Step 2: Configure TensorRT EP INT8 calibration:
    
    ```rust
    Config::default()
        .with_<module>_int8(true)
        .with_<module>_int8_calibration_table_name()
        .with_<module>_int8_use_native_calibration_table()
    ```
    
    - [Demo: ONNX Runtime quantization TensorRT](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/trt)


---


## Data Type Selection & Method Selection

- [Method Selection](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#method-selection)
- [Data Type Selection](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#data-type-selection)


## References

- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [NVIDIA Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)