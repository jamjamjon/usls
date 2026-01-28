# Getting Started Overview
<p align="center">
  <img src="https://github.com/jamjamjon/assets/releases/download/images/pipeline.png" width="800">
</p>

**usls** is a cross-platform Rust library powered by ONNX Runtime for efficient inference of SOTA vision and vision-language models (***typically under 1B parameters***).


## 🌟 Highlights

- **⚡ High Performance**: Multi-threading, SIMD, and CUDA-accelerated processing
- **🌐 Cross-Platform**: Linux, macOS, Windows with ONNX Runtime execution providers (CUDA, TensorRT, CoreML, OpenVINO, DirectML, etc.)
- **🏗️ Unified API**: Single `Model` trait inference with `run()`/`forward()`/`encode_images()`/`encode_texts()` and unified `Y` output
- **📥 Auto-Management**: Automatic model download (HuggingFace/GitHub), caching and path resolution
- **📦 Multiple Inputs**: Image, directory, video, webcam, stream and combinations
- **🎯 Precision Support**: FP32, FP16, INT8, UINT8, Q4, Q4F16, BNB4, and more
- **🛠️ Full-Stack Suite**: `DataLoader`, `Annotator`, and `Viewer` for complete workflows
- **🌱 Model Ecosystem**: 50+ SOTA vision and VLM models
