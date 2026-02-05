---
hide:
  - toc
  - navigation
---

???+ question "ONNX Runtime Issues"
    Check these issue trackers:

    - [ort](https://github.com/pykeio/ort) 
    - [onnxruntime](https://github.com/microsoft/onnxruntime)

??? danger "Linking errors with `__isoc23_strtoll`?"
    Set the ORT_DYLIB_PATH environment variable to the path to libonnxruntime.so/onnxruntime.dll
    ```bash
    export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/
    ```

    Use the dynamic loading feature:
    ```bash
    cargo run -F ort-load-dynamic --example
    ```

??? question "Other Linking Errors?"
    See [ORT Linking](https://ort.pyke.io/setup/linking) for more information.


??? question "Why no LLM models?"
    - Focus: Vision and VLM models under 1B parameters
    - LLM inference engines like vLLM already exist
    - Pure text embedding models may be added in the future

??? question "How fast is it?"
    - YOLO benchmarks: see [Performance](getting-started/overview.md#performance-reference)
    - Optimizations: multi-threading, SIMD, CUDA acceleration
    - YOLO and RFDETR are well-optimized; other models may need more work


!!! info "Still have questions?"
    Open a [GitHub Issue](https://github.com/jamjamjon/usls/issues).
