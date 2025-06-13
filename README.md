<h2 align="center">usls</h2>
<p align="center">
<a href="https://github.com/jamjamjon/usls/actions/workflows/rust-ci.yml">
        <img src="https://github.com/jamjamjon/usls/actions/workflows/rust-ci.yml/badge.svg" alt="Rust CI">
    </a>
    <a href='https://crates.io/crates/usls'>
        <img src='https://img.shields.io/crates/v/usls.svg' alt='Crates.io Version'>
    </a>
    <a href='https://github.com/microsoft/onnxruntime/releases'>
        <img src='https://img.shields.io/badge/onnxruntime-%3E%3D%201.22.0-3399FF' alt='ONNXRuntime MSRV'>
    </a>
    <a href='https://crates.io/crates/usls'>
        <img src='https://img.shields.io/crates/msrv/usls-yellow?' alt='Rust MSRV'>
    </a>
</p>

**usls** is a cross-platform Rust library powered by ONNX Runtime for efficient inference of SOTA vision and multi-modal models(typically under 1B parameters).

## 📚 Documentation
- [API Documentation](https://docs.rs/usls/latest/usls/)
- [Examples](./examples)


## 🚀 Quick Start
```bash
# CPU
cargo run -r --example yolo -- --task detect --ver 8 --scale n --dtype fp16  # q8, q4, q4f16

# NVIDIA CUDA
cargo run -r -F cuda --example yolo -- --device cuda:0   # YOLOv8-n detect by default

# NVIDIA TensorRT
cargo run -r -F tensorrt --example yolo -- --device tensorrt:0

# Apple Silicon CoreML
cargo run -r -F coreml --example yolo -- --device coreml

# Intel OpenVINO
cargo run -r -F openvino -F ort-load-dynamic --example yolo -- --device openvino:CPU

# And other EPs...
```


## ⚙️ Installation
Add the following to your `Cargo.toml`:

```toml
[dependencies]
# Recommended: Use GitHub version
usls = { git = "https://github.com/jamjamjon/usls", features = [ "cuda" ] }

# Alternative: Use crates.io version
usls = "latest-version"
```

## ⚡ Supported Models
<details>
<summary>Click to expand</summary>

| Model | Task / Description | Example |
| ----- | ----------------- | ------- |
| [BEiT](https://github.com/microsoft/unilm/tree/master/beit) | Image Classification | [demo](examples/beit) |
| [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | Image Classification | [demo](examples/convnext) |
| [FastViT](https://github.com/apple/ml-fastvit) | Image Classification | [demo](examples/fastvit) |
| [MobileOne](https://github.com/apple/ml-mobileone) | Image Classification | [demo](examples/mobileone) |
| [DeiT](https://github.com/facebookresearch/deit) | Image Classification | [demo](examples/deit) |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Vision Embedding | [demo](examples/dinov2) |
| [YOLOv5](https://github.com/ultralytics/yolov5) | Image Classification<br />Object Detection<br />Instance Segmentation | [demo](examples/yolo) |
| [YOLOv6](https://github.com/meituan/YOLOv6) | Object Detection | [demo](examples/yolo) |
| [YOLOv7](https://github.com/WongKinYiu/yolov7) | Object Detection | [demo](examples/yolo) |
| [YOLOv8<br />YOLO11](https://github.com/ultralytics/ultralytics) | Object Detection<br />Instance Segmentation<br />Image Classification<br />Oriented Object Detection<br />Keypoint Detection | [demo](examples/yolo) |
| [YOLOv9](https://github.com/WongKinYiu/yolov9) | Object Detection | [demo](examples/yolo) |
| [YOLOv10](https://github.com/THU-MIG/yolov10) | Object Detection | [demo](examples/yolo) |
| [YOLOv12](https://github.com/sunsmarterjie/yolov12) | Object Detection | [demo](examples/yolo) |
| [RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Object Detection | [demo](examples/rtdetr) |
| [RF-DETR](https://github.com/roboflow/rf-detr) | Object Detection | [demo](examples/rfdetr) |
| [PP-PicoDet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8/configs/picodet) | Object Detection | [demo](examples/picodet-layout) |
| [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) | Object Detection | [demo](examples/picodet-layout) |
| [D-FINE](https://github.com/manhbd-22022602/D-FINE) | Object Detection | [demo](examples/d-fine) |
| [DEIM](https://github.com/ShihuaHuang95/DEIM) | Object Detection | [demo](examples/deim) |
| [RTMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose) | Keypoint Detection | [demo](examples/rtmpose) |
| [DWPose](https://github.com/IDEA-Research/DWPose) | Keypoint Detection | [demo](examples/dwpose) |
| [RTMW](https://arxiv.org/abs/2407.08634) | Keypoint Detection | [demo](examples/rtmw) |
| [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) | Keypoint Detection | [demo](examples/rtmo) |
| [SAM](https://github.com/facebookresearch/segment-anything) | Segment Anything | [demo](examples/sam) |
| [SAM2](https://github.com/facebookresearch/segment-anything-2) | Segment Anything | [demo](examples/sam) |
| [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | Segment Anything | [demo](examples/sam) |
| [EdgeSAM](https://github.com/chongzhou96/EdgeSAM) | Segment Anything | [demo](examples/sam) |
| [SAM-HQ](https://github.com/SysCV/sam-hq) | Segment Anything | [demo](examples/sam) |
| [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) | Instance Segmentation | [demo](examples/yolo) |
| [YOLO-World](https://github.com/AILab-CVC/YOLO-World) | Open-Set Detection With Language | [demo](examples/yolo) |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Open-Set Detection With Language | [demo](examples/grounding-dino) |
| [CLIP](https://github.com/openai/CLIP) | Vision-Language Embedding | [demo](examples/clip) |
| [jina-clip-v1](https://huggingface.co/jinaai/jina-clip-v1) | Vision-Language Embedding | [demo](examples/clip) |
| [jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2) | Vision-Language Embedding | [demo](examples/clip) |
| [mobileclip](https://github.com/apple/ml-mobileclip) | Vision-Language Embedding | [demo](examples/clip) |
| [BLIP](https://github.com/salesforce/BLIP) | Image Captioning | [demo](examples/blip) |
| [DB(PaddleOCR-Det)](https://arxiv.org/abs/1911.08947) | Text Detection | [demo](examples/db) |
| [FAST](https://github.com/czczup/FAST) | Text Detection | [demo](examples/fast) |
| [LinkNet](https://arxiv.org/abs/1707.03718) | Text Detection | [demo](examples/linknet) |
| [SVTR(PaddleOCR-Rec)](https://arxiv.org/abs/2205.00159) | Text Recognition | [demo](examples/svtr) |
| [SLANet](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/table_recognition/algorithm_table_slanet.html) | Tabel Recognition | [demo](examples/slanet) |
| [TrOCR](https://huggingface.co/microsoft/trocr-base-printed) | Text Recognition | [demo](examples/trocr) |
| [YOLOPv2](https://arxiv.org/abs/2208.11434) | Panoptic Driving Perception | [demo](examples/yolop) |
| [DepthAnything v1<br />DepthAnything v2](https://github.com/LiheYoung/Depth-Anything) | Monocular Depth Estimation | [demo](examples/depth-anything) |
| [DepthPro](https://github.com/apple/ml-depth-pro) | Monocular Depth Estimation | [demo](examples/depth-pro) |
| [MODNet](https://github.com/ZHKKKe/MODNet) | Image Matting | [demo](examples/modnet) |
| [Sapiens](https://github.com/facebookresearch/sapiens/tree/main) | Foundation for Human Vision Models | [demo](examples/sapiens) |
| [Florence2](https://arxiv.org/abs/2311.06242) | a Variety of Vision Tasks | [demo](examples/florence2) |
| [Moondream2](https://github.com/vikhyat/moondream/tree/main) | Open-Set Object Detection<br />Open-Set Keypoints Detection<br />Image Caption<br />Visual Question Answering | [demo](examples/moondream2) |
| [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) | Open-Set Object Detection | [demo](examples/owlv2) |
| [SmolVLM(256M, 500M)](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | Visual Question Answering | [demo](examples/smolvlm) |
| [RMBG(1.4, 2.0)](https://huggingface.co/briaai/RMBG-2.0) | Image Segmentation<br />Background Removal | [demo](examples/rmbg) |
| [BEN2](https://huggingface.co/PramaLLC/BEN2) | Image Segmentation<br />Background Removal | [demo](examples/rmbg) |
| [MediaPipe: Selfie-segmentation](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter) | Image Segmentation | [demo](examples/mediapipe-selfie-segmentation) |
| [Swin2SR](https://github.com/mv-lab/swin2sr) | Image Super-Resolution and Restoration | [demo](examples/swin2sr) |
| [APISR](https://github.com/Kiteretsu77/APISR) | Real-World Anime Super-Resolution | [demo](examples/apisr) |

</details>


## 📦 Cargo Features
- **`ort-download-binaries`** (**default**): Automatically downloads prebuilt ONNXRuntime binaries for supported platforms
- **`ort-load-dynamic`**: Dynamic linking to ONNXRuntime libraries ([Guide](https://ort.pyke.io/setup/linking#dynamic-linking))
- **`video`**: Enable video stream reading and writing (via [video-rs](https://github.com/oddity-ai/video-rs) and [minifb](https://github.com/emoon/rust_minifb))
- **`cuda`**: NVIDIA CUDA GPU acceleration support
- **`tensorrt`**: NVIDIA TensorRT optimization for inference acceleration
- **`coreml`**: Apple CoreML acceleration for macOS/iOS devices
- **`openvino`**: Intel OpenVINO toolkit for CPU/GPU/VPU acceleration
- **`onednn`**: Intel oneDNN (formerly MKL-DNN) for CPU optimization
- **`directml`**: Microsoft DirectML for Windows GPU acceleration
- **`xnnpack`**: Google XNNPACK for mobile and edge device optimization
- **`rocm`**: AMD ROCm platform for GPU acceleration
- **`cann`**: Huawei CANN (Compute Architecture for Neural Networks) support
- **`rknpu`**: Rockchip NPU acceleration
- **`acl`**: Arm Compute Library for Arm processors
- **`nnapi`**: Android Neural Networks API support
- **`armnn`**: Arm NN inference engine
- **`tvm`**: Apache TVM tensor compiler stack
- **`qnn`**: Qualcomm Neural Network SDK
- **`migraphx`**: AMD MIGraphX for GPU acceleration
- **`vitis`**: Xilinx Vitis AI for FPGA acceleration
- **`azure`**: Azure Machine Learning integration


## ❓ FAQ
See [issues](https://github.com/jamjamjon/usls/issues) or open a new discussion.

## 🤝 Contributing

Contributions are welcome! If you have suggestions, bug reports, or want to add new features or models, feel free to open an issue or submit a pull request.  


## 📜 License

This project is licensed under [LICENSE](LICENSE).
