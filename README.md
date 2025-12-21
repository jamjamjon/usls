<h2 align="center">usls</h2>
<p align="center">
<a href="https://github.com/jamjamjon/usls/actions/workflows/rust-ci.yml">
        <img src="https://github.com/jamjamjon/usls/actions/workflows/rust-ci.yml/badge.svg" alt="Rust CI">
    </a>
    <a href='https://crates.io/crates/usls'>
        <img src='https://img.shields.io/crates/v/usls?logo=rust&logoColor=white' alt='Crates.io Version'>
    </a>
    <a href='https://github.com/microsoft/onnxruntime/releases'>
        <img src='https://img.shields.io/badge/onnxruntime-%3E%3D%201.22.0-3399FF?logo=onnx&logoColor=white' alt='ONNXRuntime MSRV'>
    </a>
    <a href='https://crates.io/crates/usls'>
        <img src='https://img.shields.io/crates/msrv/usls?color=yellow&logo=rust' alt='Rust MSRV'>
    </a>
</p>

**usls** is a cross-platform Rust library powered by ONNX Runtime for efficient inference of SOTA vision and vision-language models (typically under 1B parameters).

## 📚 Documentation
- [API Documentation](https://docs.rs/usls/latest/usls/)
- [Examples](./examples)


## 🚀 Quick Start

Run the **YOLO demo** to explore various YOLO-Series models with different tasks, precision, and execution providers:

- **Tasks**: `detect`, `segment`, `pose`, `classify`, `obb`
- **Versions**: `YOLOv5`, `YOLOv6`, `YOLOv7`, `YOLOv8`, `YOLOv9`, `YOLOv10`, `YOLO11`, `YOLOv12`, `YOLOv13`
- **Scales**: `n`, `s`, `m`, `l`, `x`
- **Precision**: `fp32`, `fp16`, `q8`, `q4`, `q4f16`, `bnb4`
- **Execution Providers**: `CPU`, `CUDA`, `TensorRT`, `CoreML`, `OpenVINO`, and more

```bash
# CPU: Object detection, YOLOv8n, FP16
cargo run -r --example yolo -- --task detect --ver 8 --scale n --dtype fp16

# NVIDIA CUDA: Instance segmentation, YOLO11m
cargo run -r -F cuda --example yolo -- --task segment --ver 11 --scale m --device cuda:0

# NVIDIA TensorRT
cargo run -r -F tensorrt --example yolo -- --device tensorrt:0

# Apple Silicon CoreML
cargo run -r -F coreml --example yolo -- --device coreml

# Intel OpenVINO: CPU/GPU/VPU acceleration
cargo run -r -F openvino -F ort-load-dynamic --example yolo -- --device openvino:CPU

# Show all available options
cargo run -r --example yolo -- --help
```

See [YOLO Examples](./examples/yolo/README.md) for more details and use cases.


## ⚙️ Installation
Add the following to your `Cargo.toml`:

```toml
[dependencies]
# Use GitHub version
usls = { git = "https://github.com/jamjamjon/usls", features = [ "cuda" ] }

# Alternative: Use crates.io version
usls = { version = "latest-version", features = [ "cuda" ] }
```

## 📦 Cargo Features

> ❕ Features in ***italics*** are enabled by default.

- ### Runtime & Utilities
  - ***`ort-download-binaries`***: Auto-download ONNX Runtime binaries from [pyke](https://ort.pyke.io/perf/execution-providers).
  - **`ort-load-dynamic`**: Linking ONNX Runtime by your self. Use this if `pyke` doesn't provide prebuilt binaries for your platform or you want to link your local ONNX Runtime library. See [Linking Guide](https://ort.pyke.io/setup/linking#static-linking) for more details.
  - **`viewer`**: Image/video visualization ([minifb](https://github.com/emoon/rust_minifb)). Similar to OpenCV `imshow()`. See [example](./examples/imshow.rs).
  - **`video`**: Video I/O support ([video-rs](https://github.com/oddity-ai/video-rs)). Enable this to read/write video streams. See [example](./examples/read_video.rs)
  - **`hf-hub`**: Hugging Face Hub support for downloading models from Hugging Face repositories.
  - **`tokenizers`**: Tokenizer support for vision-language models. Automatically enabled when using vision-language model features (blip, clip, florence2, grounding-dino, fastvlm, moondream2, owl, smolvlm, trocr, yoloe).
  - **`slsl`**: SLSL tensor library support. Automatically enabled when using `yolo` or `clip` features.

- ### Execution Providers
  Hardware acceleration for inference. 

  - **`cuda`**, **`tensorrt`**: NVIDIA GPU acceleration
  - **`coreml`**: Apple Silicon acceleration
  - **`openvino`**: Intel CPU/GPU/VPU acceleration
  - **`onednn`**, **`directml`**, **`xnnpack`**, **`rocm`**, **`cann`**, **`rknpu`**, **`acl`**, **`nnapi`**, **`armnn`**, **`tvm`**, **`qnn`**, **`migraphx`**, **`vitis`**, **`azure`**: Various hardware/platform support

  See [ONNX Runtime docs](https://onnxruntime.ai/docs/execution-providers/) and [ORT performance guide](https://ort.pyke.io/perf/execution-providers) for details.

- ### Model Selection
  Almost each model is a separate feature. Enable only what you need to reduce compile time and binary size.

  - *`yolo`*, `sam`, `clip`, `image-classifier`, `dino`, `rtmpose`, `rtdetr`, `db`, ...
  - **All models**: `all-models` (enables all model features)

  See [Supported Models](#-supported-models) for the complete list with feature names.


## ⚡ Supported Models


<details>
<summary><b> 👀 View all models (Click to expand)</b></summary>

| Model | Task / Description | Feature | Example |
| ----- | ----------------- | ------- | ------- |
| [BEiT](https://github.com/microsoft/unilm/tree/master/beit) | Image Classification | `image-classifier` | [demo](examples/beit) |
| [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | Image Classification | `image-classifier` | [demo](examples/convnext) |
| [FastViT](https://github.com/apple/ml-fastvit) | Image Classification | `image-classifier` | [demo](examples/fastvit) |
| [MobileOne](https://github.com/apple/ml-mobileone) | Image Classification | `image-classifier` | [demo](examples/mobileone) |
| [DeiT](https://github.com/facebookresearch/deit) | Image Classification | `image-classifier` | [demo](examples/deit) |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Vision Embedding | `dino` | [demo](examples/dinov2) |
| [DINOv3](https://github.com/facebookresearch/dinov3) | Vision Embedding | `dino` | [demo](examples/dinov3) |
| [YOLOv5](https://github.com/ultralytics/yolov5) | Image Classification<br />Object Detection<br />Instance Segmentation | `yolo` | [demo](examples/yolo) |
| [YOLOv6](https://github.com/meituan/YOLOv6) | Object Detection | `yolo` | [demo](examples/yolo) |
| [YOLOv7](https://github.com/WongKinYiu/yolov7) | Object Detection | `yolo` | [demo](examples/yolo) |
| [YOLOv8<br />YOLO11](https://github.com/ultralytics/ultralytics) | Object Detection<br />Instance Segmentation<br />Image Classification<br />Oriented Object Detection<br />Keypoint Detection | `yolo` | [demo](examples/yolo) |
| [YOLOv9](https://github.com/WongKinYiu/yolov9) | Object Detection | `yolo` | [demo](examples/yolo) |
| [YOLOv10](https://github.com/THU-MIG/yolov10) | Object Detection | `yolo` | [demo](examples/yolo) |
| [YOLOv12](https://github.com/sunsmarterjie/yolov12) | Image Classification<br />Object Detection<br />Instance Segmentation | `yolo` | [demo](examples/yolo) |
| [YOLOv13](https://github.com/iMoonLab/yolov13) | Object Detection | `yolo` | [demo](examples/yolo) |
| [RT-DETRv1, v2](https://github.com/lyuwenyu/RT-DETR) | Object Detection | `rtdetr` | [demo](examples/rtdetr) |
| [RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | Object Detection | `rtdetr` | [demo](examples/rtdetr) |
| [RF-DETR](https://github.com/roboflow/rf-detr) | Object Detection | `rfdetr` | [demo](examples/rfdetr) |
| [PP-PicoDet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8/configs/picodet) | Object Detection | `picodet` | [demo](examples/picodet-layout) |
| [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) | Object Detection | `picodet` | [demo](examples/picodet-layout) |
| [D-FINE](https://github.com/manhbd-22022602/D-FINE) | Object Detection | `rtdetr` | [demo](examples/d-fine) |
| [DEIM](https://github.com/ShihuaHuang95/DEIM) | Object Detection | `rtdetr` | [demo](examples/deim) |
| [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | Object Detection | `rtdetr` | [demo](examples/deimv2) |
| [RTMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose) | Keypoint Detection | `rtmpose` | [demo](examples/rtmpose) |
| [DWPose](https://github.com/IDEA-Research/DWPose) | Keypoint Detection | `rtmpose` | [demo](examples/dwpose) |
| [RTMW](https://arxiv.org/abs/2407.08634) | Keypoint Detection | `rtmpose` | [demo](examples/rtmw) |
| [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) | Keypoint Detection | `rtmo` | [demo](examples/rtmo) |
| [SAM](https://github.com/facebookresearch/segment-anything) | Segment Anything | `sam` | [demo](examples/sam) |
| [SAM2](https://github.com/facebookresearch/segment-anything-2) | Segment Anything | `sam2` | [demo](examples/sam2) |
| [SAM3](https://github.com/facebookresearch/segment-anything-3) | Segment Anything | `sam3` | [demo](examples/sam3) |
| [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | Segment Anything | `sam` | [demo](examples/sam) |
| [EdgeSAM](https://github.com/chongzhou96/EdgeSAM) | Segment Anything | `sam` | [demo](examples/sam) |
| [SAM-HQ](https://github.com/SysCV/sam-hq) | Segment Anything | `sam` | [demo](examples/sam) |
| [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) | Instance Segmentation | `yolo` | [demo](examples/yolo) |
| [YOLO-World](https://github.com/AILab-CVC/YOLO-World) | Open-Set Detection With Language | `yolo` | [demo](examples/yolo) |
| [YOLOE](https://github.com/THU-MIG/yoloe) | Open-Set Detection And Segmentation | `yoloe` | [demo-prompt-free](examples/yoloe-prompt-free)<br />[demo-prompt(visual & textual)](examples/yoloe-prompt) |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Open-Set Detection With Language | `grounding-dino` | [demo](examples/grounding-dino) |
| [MM-GDINO](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md) | Open-Set Detection With Language | `grounding-dino` | [demo](examples/grounding-dino) |
| [LLMDet](https://github.com/iSEE-Laboratory/LLMDet) | Open-Set Detection With Language | `grounding-dino` | [demo](examples/grounding-dino) |
| [CLIP](https://github.com/openai/CLIP) | Vision-Language Embedding | `clip` | [demo](examples/clip) |
| [jina-clip-v1](https://huggingface.co/jinaai/jina-clip-v1) | Vision-Language Embedding | `clip` | [demo](examples/clip) |
| [jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2) | Vision-Language Embedding | `clip` | [demo](examples/clip) |
| [mobileclip & mobileclip2](https://github.com/apple/ml-mobileclip) | Vision-Language Embedding | `clip` | [demo](examples/clip) |
| [BLIP](https://github.com/salesforce/BLIP) | Image Captioning | `blip` | [demo](examples/blip) |
| [DB(PaddleOCR-Det)](https://arxiv.org/abs/1911.08947) | Text Detection | `db` | [demo](examples/db) |
| [FAST](https://github.com/czczup/FAST) | Text Detection | `db` | [demo](examples/fast) |
| [LinkNet](https://arxiv.org/abs/1707.03718) | Text Detection | `db` | [demo](examples/linknet) |
| [SVTR(PaddleOCR-Rec)](https://arxiv.org/abs/2205.00159) | Text Recognition | `svtr` | [demo](examples/svtr) |
| [SLANet](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/table_recognition/algorithm_table_slanet.html) | Tabel Recognition | `slanet` | [demo](examples/slanet) |
| [TrOCR](https://huggingface.co/microsoft/trocr-base-printed) | Text Recognition | `trocr` | [demo](examples/trocr) |
| [YOLOPv2](https://arxiv.org/abs/2208.11434) | Panoptic Driving Perception | `yolop` | [demo](examples/yolop) |
| [DepthAnything v1<br />DepthAnything v2](https://github.com/LiheYoung/Depth-Anything) | Monocular Depth Estimation | `depth-anything` | [demo](examples/depth-anything) |
| [DepthPro](https://github.com/apple/ml-depth-pro) | Monocular Depth Estimation | `depth-pro` | [demo](examples/depth-pro) |
| [MODNet](https://github.com/ZHKKKe/MODNet) | Image Matting | `modnet` | [demo](examples/modnet) |
| [Sapiens](https://github.com/facebookresearch/sapiens/tree/main) | Foundation for Human Vision Models | `sapiens` | [demo](examples/sapiens) |
| [Florence2](https://arxiv.org/abs/2311.06242) | A Variety of Vision Tasks | `florence2` | [demo](examples/florence2) |
| [Moondream2](https://github.com/vikhyat/moondream/tree/main) | Open-Set Object Detection<br />Open-Set Keypoints Detection<br />Image Caption<br />Visual Question Answering | `moondream2` | [demo](examples/moondream2) |
| [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) | Open-Set Object Detection | `owl` | [demo](examples/owlv2) |
| [SmolVLM(256M, 500M)](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | Visual Question Answering | `smolvlm` | [demo](examples/smolvlm) |
| [FastVLM(0.5B)](https://github.com/apple/ml-fastvlm) | Vision Language Models | `fastvlm` | [demo](examples/fastvlm) |
| [RMBG(1.4, 2.0)](https://huggingface.co/briaai/RMBG-2.0) | Image Segmentation<br />Background Removal | `rmbg` | [demo](examples/rmbg) |
| [BEN2](https://huggingface.co/PramaLLC/BEN2) | Image Segmentation<br />Background Removal | `ben2` | [demo](examples/rmbg) |
| [MediaPipe: Selfie-segmentation](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter) | Image Segmentation | `mediapipe-segmenter` | [demo](examples/mediapipe-selfie-segmentation) |
| [Swin2SR](https://github.com/mv-lab/swin2sr) | Image Super-Resolution and Restoration | `swin2sr` | [demo](examples/swin2sr) |
| [APISR](https://github.com/Kiteretsu77/APISR) | Real-World Anime Super-Resolution | `apisr` | [demo](examples/apisr) |
| [RAM & RAM++](https://github.com/xinyu1205/recognize-anything) | Image Tagging | `ram` | [demo](examples/ram) |

</details>


## ❓ FAQ
See [issues](https://github.com/jamjamjon/usls/issues) or open a new discussion.

## 🤝 Contributing

Contributions are welcome! If you have suggestions, bug reports, or want to add new features or models, feel free to open an issue or submit a pull request.

## 🙏 Acknowledgments

This project is built on top of [ort (ONNX Runtime for Rust)](https://github.com/pykeio/ort), which provides seamless Rust bindings for [ONNX Runtime](https://github.com/microsoft/onnxruntime). Special thanks to the `ort` maintainers.

Thanks to all the open-source libraries and their maintainers that make this project possible. See [Cargo.toml](Cargo.toml) for a complete list of dependencies.

## 📜 License

This project is licensed under [LICENSE](LICENSE).