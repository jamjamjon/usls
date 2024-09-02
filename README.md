<p align="center">
    <h2 align="center">usls</h2>
</p>

<p align="center">
    | <a href="https://docs.rs/usls"><strong>Documentation</strong></a> |
    <br>
    <br>
    <a href='https://github.com/microsoft/onnxruntime/releases'>
      <img src='https://img.shields.io/badge/ONNXRuntime-v1.19.x-239DFF?style=for-the-badge&logo=onnx' alt='ONNXRuntime Release Page'>
    </a>
    <a href='https://developer.nvidia.com/cuda-toolkit-archive'>
      <img src='https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia' alt='CUDA Toolkit Page'>
    </a>
    <a href='https://developer.nvidia.com/tensorrt'>
      <img src='https://img.shields.io/badge/TensorRT-10.x.x.x-76B900?style=for-the-badge&logo=nvidia' alt='TensorRT Page'>
    </a>
</p>

<p align="center">
   <a href='https://crates.io/crates/usls'>
      <img src='https://img.shields.io/crates/v/usls.svg?style=for-the-badge&logo=rust' alt='Crates Page'>
   </a>
   <!-- Documentation Badge -->
<!--    <a href="https://docs.rs/usls">
      <img src='https://img.shields.io/badge/Documents-usls-000000?style=for-the-badge&logo=docs.rs' alt='Documentation'>
   </a> -->
   <!-- Downloads Badge -->
   <a href="">
       <img alt="Crates.io Total Downloads" src="https://img.shields.io/crates/d/usls?style=for-the-badge&color=3ECC5F">
   </a>
    
</p>

**`usls`** is a Rust library integrated with **ONNXRuntime** that provides a collection of state-of-the-art models for **Computer Vision** and **Vision-Language** tasks, including:

- **YOLO Models**: [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv9](https://github.com/WongKinYiu/yolov9), [YOLOv10](https://github.com/THU-MIG/yolov10)
- **SAM Models**: [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/segment-anything-2), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [EdgeSAM](https://github.com/chongzhou96/EdgeSAM), [SAM-HQ](https://github.com/SysCV/sam-hq), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
- **Vision Models**: [RTDETR](https://arxiv.org/abs/2304.08069), [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo), [DB](https://arxiv.org/abs/1911.08947), [SVTR](https://arxiv.org/abs/2205.00159), [Depth-Anything-v1-v2](https://github.com/LiheYoung/Depth-Anything), [DINOv2](https://github.com/facebookresearch/dinov2), [MODNet](https://github.com/ZHKKKe/MODNet), [Sapiens](https://arxiv.org/abs/2408.12569)
- **Vision-Language Models**: [CLIP](https://github.com/openai/CLIP), [BLIP](https://arxiv.org/abs/2201.12086), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [YOLO-World](https://github.com/AILab-CVC/YOLO-World)

<details>
<summary>Click to expand Supported Models</summary>

## Supported Models

| Model                                                               | Task / Type                                                                                   | Example                    | CUDA f32 | CUDA f16 | TensorRT f32 | TensorRT f16 |
|---------------------------------------------------------------------|----------------------------------------------------------------------------------------------|----------------------------|----------|----------|--------------|--------------|
| [YOLOv5](https://github.com/ultralytics/yolov5)                    | Classification<br>Object Detection<br>Instance Segmentation                                       | [demo](examples/yolo)      | ✅       | ✅       | ✅           | ✅           |
| [YOLOv6](https://github.com/meituan/YOLOv6)                        | Object Detection                                                                             | [demo](examples/yolo)      | ✅       | ✅       | ✅           | ✅           |
| [YOLOv7](https://github.com/WongKinYiu/yolov7)                     | Object Detection                                                                             | [demo](examples/yolo)      | ✅       | ✅       | ✅           | ✅           |
| [YOLOv8](https://github.com/ultralytics/ultralytics)                | Object Detection<br>Instance Segmentation<br>Classification<br>Oriented Object Detection<br>Keypoint Detection | [demo](examples/yolo)      | ✅       | ✅       | ✅           | ✅           |
| [YOLOv9](https://github.com/WongKinYiu/yolov9)                     | Object Detection                                                                             | [demo](examples/yolo)      | ✅       | ✅       | ✅           | ✅           |
| [YOLOv10](https://github.com/THU-MIG/yolov10)                      | Object Detection                                                                             | [demo](examples/yolo)      | ✅       | ✅       | ✅           | ✅           |
| [RTDETR](https://arxiv.org/abs/2304.08069)                         | Object Detection                                                                             | [demo](examples/yolo)      | ✅       | ✅       | ✅           | ✅           |
| [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)                 | Instance Segmentation                                                                         | [demo](examples/yolo)      | ✅       | ✅       | ✅           | ✅           |
| [SAM](https://github.com/facebookresearch/segment-anything)         | Segment Anything                                                                             | [demo](examples/sam)       | ✅       | ✅       |              |              |
| [SAM2](https://github.com/facebookresearch/segment-anything-2)      | Segment Anything                                                                             | [demo](examples/sam)       | ✅       | ✅       |              |              |
| [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)             | Segment Anything                                                                             | [demo](examples/sam)       | ✅       | ✅       |              |              |
| [EdgeSAM](https://github.com/chongzhou96/EdgeSAM)                  | Segment Anything                                                                             | [demo](examples/sam)       | ✅       | ✅       |              |              |
| [SAM-HQ](https://github.com/SysCV/sam-hq)                          | Segment Anything                                                                             | [demo](examples/sam)       | ✅       | ✅       |              |              |
| [YOLO-World](https://github.com/AILab-CVC/YOLO-World)               | Object Detection                                                                             | [demo](examples/yolo)      | ✅       | ✅       | ✅           | ✅           |
| [DINOv2](https://github.com/facebookresearch/dinov2)               | Vision-Self-Supervised                                                                        | [demo](examples/dinov2)    | ✅       | ✅       | ✅           | ✅           |
| [CLIP](https://github.com/openai/CLIP)                             | Vision-Language                                                                             | [demo](examples/clip)      | ✅       | ✅       | ✅ Visual<br>❌ Textual | ✅ Visual<br>❌ Textual |
| [BLIP](https://github.com/salesforce/BLIP)                         | Vision-Language                                                                             | [demo](examples/blip)      | ✅       | ✅       | ✅ Visual<br>❌ Textual | ✅ Visual<br>❌ Textual |
| [DB](https://arxiv.org/abs/1911.08947)                             | Text Detection                                                                               | [demo](examples/db)        | ✅       | ✅       | ✅           | ✅           |
| [SVTR](https://arxiv.org/abs/2205.00159)                           | Text Recognition                                                                            | [demo](examples/svtr)      | ✅       | ✅       | ✅           | ✅           |
| [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) | Keypoint Detection                                                                          | [demo](examples/rtmo)      | ✅       | ✅       | ❌           | ❌           |
| [YOLOPv2](https://arxiv.org/abs/2208.11434)                        | Panoptic Driving Perception                                                                   | [demo](examples/yolop)     | ✅       | ✅       | ✅           | ✅           |
| [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)      | Monocular Depth Estimation                                                                    | [demo](examples/depth-anything) | ✅       | ✅       | ❌           | ❌           |
| [MODNet](https://github.com/ZHKKKe/MODNet)                         | Image Matting                                                                               | [demo](examples/modnet)    | ✅       | ✅       | ✅           | ✅           |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)   | Open-Set Detection With Language                                                             | [demo](examples/grounding-dino) | ✅       | ✅       |              |              |
| [Sapiens](https://github.com/facebookresearch/sapiens/tree/main)   | Body Part Segmentation                                   | [demo](examples/sapiens) | ✅       | ✅       |              |              |

</details>


## ⛳️ ONNXRuntime Linking 

You have two options to link the ONNXRuntime library

- ### Option 1: Manual Linking

    - #### For detailed setup instructions, refer to the [ORT documentation](https://ort.pyke.io/setup/linking).

    - #### For Linux or macOS Users:
        - Download the ONNX Runtime package from the [Releases page](https://github.com/microsoft/onnxruntime/releases).
        - Set up the library path by exporting the `ORT_DYLIB_PATH` environment variable:
           ```shell
           export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so.1.19.0
           ```
       
- ### Option 2: Automatic Download
  Just use `--features auto`
  ```shell
  cargo run -r --example yolo --features auto
  ```



## 🎈 Quick Start

```Shell
cargo run -r --example yolo   # blip, clip, yolop, svtr, db, ...
```

## 🥂 Integrate Into Your Own Project

Add `usls` as a dependency to your project's `Cargo.toml`
```Shell
cargo add usls
```

Or use a specific commit:
```Toml
[dependencies]
usls = { git = "https://github.com/jamjamjon/usls", rev = "commit-sha" }
```

## 📌 License
This project is licensed under [LICENSE](LICENSE).
