<h2 align="center">usls</h2>

<p align="center">
    <a href="https://github.com/jamjamjon/usls/actions/workflows/rust-ci.yml">
        <img src="https://github.com/jamjamjon/usls/actions/workflows/rust-ci.yml/badge.svg" alt="Rust Continuous Integration Badge">
    </a>
    <a href='https://crates.io/crates/usls'>
        <img src='https://img.shields.io/crates/v/usls.svg' alt='usls Version'>
    </a>
    <a href='https://crates.io/crates/usls'>
        <img src='https://img.shields.io/crates/msrv/usls-yellow?' alt='Rust MSRV'>
    </a>
    <a href='https://github.com/microsoft/onnxruntime/releases'>
        <img src='https://img.shields.io/badge/onnxruntime-%3E%3D%201.19.0-3399FF' alt='ONNXRuntime MSRV'>
    </a>
    <a href='https://developer.nvidia.com/cuda-toolkit-archive'>
        <img src='https://img.shields.io/badge/cuda-%3E%3D%2012.0-green' alt='CUDA MSRV'>
    </a>
    <a href='https://developer.nvidia.com/tensorrt'>
        <img src='https://img.shields.io/badge/TensorRT-%3E%3D%2012.0-0ABF53' alt='TensorRT MSRV'>
    </a>
    <a href="https://crates.io/crates/usls">
        <img alt="Crates.io Total Downloads" src="https://img.shields.io/crates/d/usls?&color=946CE6">
    </a>
</p>
<p align="center">
    <a href="./examples">
        <img src="https://img.shields.io/badge/Examples-1A86FD?&logo=anki" alt="Examples">
    </a>
    <a href='https://docs.rs/usls'>
        <img src='https://img.shields.io/badge/Docs-usls-yellow?&logo=docs.rs&color=FFA200' alt='usls documentation'>
    </a>
</p>

**usls** is a Rust library integrated with  **ONNXRuntime**, offering a suite of advanced models for **Computer Vision** and **Vision-Language** tasks, including:

- **YOLO Models**: [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv9](https://github.com/WongKinYiu/yolov9), [YOLOv10](https://github.com/THU-MIG/yolov10), [YOLO11](https://github.com/ultralytics/ultralytics), [YOLOv12](https://github.com/sunsmarterjie/yolov12)
- **SAM Models**: [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/segment-anything-2), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [EdgeSAM](https://github.com/chongzhou96/EdgeSAM), [SAM-HQ](https://github.com/SysCV/sam-hq), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
- **Vision Models**: [RT-DETR](https://arxiv.org/abs/2304.08069), [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo), [Depth-Anything](https://github.com/LiheYoung/Depth-Anything), [DINOv2](https://github.com/facebookresearch/dinov2), [MODNet](https://github.com/ZHKKKe/MODNet), [Sapiens](https://arxiv.org/abs/2408.12569), [DepthPro](https://github.com/apple/ml-depth-pro), [FastViT](https://github.com/apple/ml-fastvit), [BEiT](https://github.com/microsoft/unilm/tree/master/beit), [MobileOne](https://github.com/apple/ml-mobileone)
- **Vision-Language Models**: [CLIP](https://github.com/openai/CLIP), [jina-clip-v1](https://huggingface.co/jinaai/jina-clip-v1), [BLIP](https://arxiv.org/abs/2201.12086), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [Florence2](https://arxiv.org/abs/2311.06242), [Moondream2](https://github.com/vikhyat/moondream/tree/main)
- **OCR Models**: [FAST](https://github.com/czczup/FAST), [DB(PaddleOCR-Det)](https://arxiv.org/abs/1911.08947), [SVTR(PaddleOCR-Rec)](https://arxiv.org/abs/2205.00159), [SLANet](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/table_recognition/algorithm_table_slanet.html), [TrOCR](https://huggingface.co/microsoft/trocr-base-printed), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

<details>
<summary>👉 More Supported Models</summary>

| Model                                                                                                          | Task / Description                                                                                                           | Example                      | CoreML | CUDA<br />FP32 | CUDA<br />FP16 | TensorRT<br />FP32 | TensorRT<br />FP16 |
| -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | ------ | -------------- | -------------- | ------------------ | ------------------ |
| [BEiT](https://github.com/microsoft/unilm/tree/master/beit)                                                       | Image Classification                                                                                                         | [demo](examples/beit)           | ✅     | ✅             | ✅             |                    |                    |
| [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)                                                          | Image Classification                                                                                                         | [demo](examples/convnext)       | ✅     | ✅             | ✅             |                    |                    |
| [FastViT](https://github.com/apple/ml-fastvit)                                                                    | Image Classification                                                                                                         | [demo](examples/fastvit)        | ✅     | ✅             | ✅             |                    |                    |
| [MobileOne](https://github.com/apple/ml-mobileone)                                                                | Image Classification                                                                                                         | [demo](examples/mobileone)      | ✅     | ✅             | ✅             |                    |                    |
| [DeiT](https://github.com/facebookresearch/deit)                                                                  | Image Classification                                                                                                         | [demo](examples/deit)           | ✅     | ✅             | ✅             |                    |                    |
| [DINOv2](https://github.com/facebookresearch/dinov2)                                                              | Vision Embedding                                                                                                            | [demo](examples/dinov2)         | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [YOLOv5](https://github.com/ultralytics/yolov5)                                                                   | Image Classification<br />Object Detection<br />Instance Segmentation                                                        | [demo](examples/yolo)           | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [YOLOv6](https://github.com/meituan/YOLOv6)                                                                       | Object Detection                                                                                                             | [demo](examples/yolo)           | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [YOLOv7](https://github.com/WongKinYiu/yolov7)                                                                    | Object Detection                                                                                                             | [demo](examples/yolo)           | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [YOLOv8<br />YOLO11](https://github.com/ultralytics/ultralytics)                                                  | Object Detection<br />Instance Segmentation<br />Image Classification<br />Oriented Object Detection<br />Keypoint Detection | [demo](examples/yolo)           | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [YOLOv9](https://github.com/WongKinYiu/yolov9)                                                                    | Object Detection                                                                                                             | [demo](examples/yolo)           | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [YOLOv10](https://github.com/THU-MIG/yolov10)                                                                     | Object Detection                                                                                                             | [demo](examples/yolo)           | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [YOLOv12](https://github.com/sunsmarterjie/yolov12)                                                                     | Object Detection                                                                                                             | [demo](examples/yolo)           | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [RT-DETR](https://github.com/lyuwenyu/RT-DETR)                                                                    | Object Detection                                                                                                             | [demo](examples/rtdetr)         | ✅     | ✅             | ✅             |                    |                    |
| [PP-PicoDet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8/configs/picodet)                    | Object Detection                                                                                                             | [demo](examples/picodet-layout) | ✅     | ✅             | ✅             |                    |                    |
| [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)                                                   | Object Detection                                                                                                             | [demo](examples/picodet-layout) | ✅     | ✅             | ✅             |                    |                    |
| [D-FINE](https://github.com/manhbd-22022602/D-FINE)                                                               | Object Detection                                                                                                             | [demo](examples/d-fine)         | ✅     | ✅             | ✅             |                    |                    |
| [DEIM](https://github.com/ShihuaHuang95/DEIM)                                                                     | Object Detection                                                                                                             | [demo](examples/deim)           | ✅     | ✅             | ✅             |                    |                    |
| [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo)                                              | Keypoint Detection                                                                                                           | [demo](examples/rtmo)           | ✅     | ✅             | ✅             | ❌                 | ❌                 |
| [SAM](https://github.com/facebookresearch/segment-anything)                                                       | Segment Anything                                                                                                             | [demo](examples/sam)            | ✅     | ✅             | ✅             |                    |                    |
| [SAM2](https://github.com/facebookresearch/segment-anything-2)                                                    | Segment Anything                                                                                                             | [demo](examples/sam)            | ✅     | ✅             | ✅             |                    |                    |
| [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)                                                           | Segment Anything                                                                                                             | [demo](examples/sam)            | ✅     | ✅             | ✅             |                    |                    |
| [EdgeSAM](https://github.com/chongzhou96/EdgeSAM)                                                                 | Segment Anything                                                                                                             | [demo](examples/sam)            | ✅     | ✅             | ✅             |                    |                    |
| [SAM-HQ](https://github.com/SysCV/sam-hq)                                                                         | Segment Anything                                                                                                             | [demo](examples/sam)            | ✅     | ✅             | ✅             |                    |                    |
| [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)                                                               | Instance Segmentation                                                                                                        | [demo](examples/yolo)           | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [YOLO-World](https://github.com/AILab-CVC/YOLO-World)                                                             | Open-Set Detection With Language                                                                                             | [demo](examples/yolo)           | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)                                                   | Open-Set Detection With Language                                                                                             | [demo](examples/grounding-dino) | ✅     | ✅             | ✅             |                    |                    |
| [CLIP](https://github.com/openai/CLIP)                                                                            | Vision-Language Embedding                                                                                                    | [demo](examples/clip)           | ✅     | ✅             | ✅             | ❌                 | ❌                 |
| [jina-clip-v1](https://huggingface.co/jinaai/jina-clip-v1)                                                        | Vision-Language Embedding                                                                                                    | [demo](examples/clip)           | ✅     | ✅             | ✅             | ❌                 | ❌                 |
| [BLIP](https://github.com/salesforce/BLIP)                                                                        | Image Captioning                                                                                                             | [demo](examples/blip)           | ✅     | ✅             | ✅             | ❌                 | ❌                 |
| [DB(PaddleOCR-Det)](https://arxiv.org/abs/1911.08947)                                                             | Text Detection                                                                                                               | [demo](examples/db)             | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [FAST](https://github.com/czczup/FAST)                                                                            | Text Detection                                                                                                               | [demo](examples/fast)           | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [LinkNet](https://arxiv.org/abs/1707.03718)                                                                       | Text Detection                                                                                                               | [demo](examples/linknet)        | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [SVTR(PaddleOCR-Rec)](https://arxiv.org/abs/2205.00159)                                                           | Text Recognition                                                                                                             | [demo](examples/svtr)           | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [SLANet](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/table_recognition/algorithm_table_slanet.html) | Tabel Recognition                                                                                                            | [demo](examples/slanet)         | ✅     | ✅             | ✅             |                    |                    |
| [TrOCR](https://huggingface.co/microsoft/trocr-base-printed)                                                      | Text Recognition                                                                                                             | [demo](examples/trocr)          | ✅     | ✅             | ✅             |                    |                    |
| [YOLOPv2](https://arxiv.org/abs/2208.11434)                                                                       | Panoptic Driving Perception                                                                                                  | [demo](examples/yolop)          | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [DepthAnything v1<br />DepthAnything v2](https://github.com/LiheYoung/Depth-Anything)                             | Monocular Depth Estimation                                                                                                   | [demo](examples/depth-anything) | ✅     | ✅             | ✅             | ❌                 | ❌                 |
| [DepthPro](https://github.com/apple/ml-depth-pro)                                                                 | Monocular Depth Estimation                                                                                                   | [demo](examples/depth-pro)      | ✅     | ✅             | ✅             |                    |                    |
| [MODNet](https://github.com/ZHKKKe/MODNet)                                                                        | Image Matting                                                                                                                | [demo](examples/modnet)         | ✅     | ✅             | ✅             | ✅                 | ✅                 |
| [Sapiens](https://github.com/facebookresearch/sapiens/tree/main)                                                  | Foundation for Human Vision Models                                                                                           | [demo](examples/sapiens)        | ✅     | ✅             | ✅             |                    |                    |
| [Florence2](https://arxiv.org/abs/2311.06242)                                                                     | a Variety of Vision Tasks                                                                                                    | [demo](examples/florence2)      | ✅     | ✅             | ✅             |                    |                    |
| [Moondream2](https://github.com/vikhyat/moondream/tree/main)                                                      | Open-Set Object Detection<br />Open-Set Keypoints Detection<br />Image Caption<br />Visual Question Answering               | [demo](examples/moondream2)     | ✅     | ✅             | ✅             |                    |                    |
| [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)                                                | Open-Set Object Detection                                                                                                    | [demo](examples/owlv2)          | ✅     | ✅             | ✅             |                    |                    |
| [SmolVLM(256M, 500M)](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)                                                | Visual Question Answering                                                                                                    | [demo](examples/smolvlm)          | ✅     | ✅             | ✅             |                    |                    |

</details>

## ⛳️ Cargo Features

By default, **none of the following features are enabled**. You can enable them as needed:

- **`auto`**: Automatically downloads prebuilt ONNXRuntime binaries from Pyke’s CDN for supported platforms.

  - If disabled, you'll need to [compile `ONNXRuntime` from source](https://github.com/microsoft/onnxruntime) or [download a precompiled package](https://github.com/microsoft/onnxruntime/releases), and then [link it manually](https://ort.pyke.io/setup/linking).

    <details>
    <summary>👉 For Linux or macOS Users</summary>

    - Download from the [Releases page](https://github.com/microsoft/onnxruntime/releases).
    - Set up the library path by exporting the `ORT_DYLIB_PATH` environment variable:
      ```shell
      export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so.1.20.1
      ```

    </details>
- **`ffmpeg`**: Adds support for video streams, real-time frame visualization, and video export.

  - Powered by [video-rs](https://github.com/oddity-ai/video-rs) and [minifb](https://github.com/emoon/rust_minifb). For any issues related to `ffmpeg` features, please refer to the issues of these two crates.
- **`cuda`**: Enables the NVIDIA TensorRT provider.
- **`trt`**: Enables the NVIDIA TensorRT provider.
- **`mps`**: Enables the Apple CoreML provider.

## 🎈 Example

* **Using `CUDA`**

  ```
  cargo run -r -F cuda --example yolo -- --device cuda:0
  ```
* **Using Apple `CoreML`**

  ```
  cargo run -r -F mps --example yolo -- --device mps
  ```
* **Using `TensorRT`**

  ```
  cargo run -r -F trt --example yolo -- --device trt
  ```
* **Using `CPU`**

  ```
  cargo run -r --example yolo
  ```

All examples are located in the [examples](./examples/) directory.

## 🥂 Integrate Into Your Own Project

Add `usls` as a dependency to your project's `Cargo.toml`

```Shell
cargo add usls -F cuda
```

Or use a specific commit:

```Toml
[dependencies]
usls = { git = "https://github.com/jamjamjon/usls", rev = "commit-sha" }
```

## 🥳 If you find this helpful, please give it a star ⭐

## 📌 License

This project is licensed under [LICENSE](LICENSE).
