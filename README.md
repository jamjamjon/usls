# usls

[![Static Badge](https://img.shields.io/crates/v/usls.svg?style=for-the-badge&logo=rust)](https://crates.io/crates/usls) [![Static Badge](https://img.shields.io/badge/ONNXRuntime-v1.17.x-yellow?style=for-the-badge&logo=docs.rs)](https://github.com/microsoft/onnxruntime/releases) [![Static Badge](https://img.shields.io/badge/CUDA-11.x-green?style=for-the-badge&logo=docs.rs)](https://developer.nvidia.com/cuda-toolkit-archive) [![Static Badge](https://img.shields.io/badge/TRT-8.6.x.x-blue?style=for-the-badge&logo=docs.rs)](https://developer.nvidia.com/tensorrt)  
[![Static Badge](https://img.shields.io/badge/Documents-usls-blue?style=for-the-badge&logo=docs.rs)](https://docs.rs/usls) ![Static Badge](https://img.shields.io/crates/d/usls?style=for-the-badge)



A Rust library integrated with **ONNXRuntime**, providing a collection of **Computer Vison** and **Vision-Language** models including [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv9](https://github.com/WongKinYiu/yolov9), [YOLOv10](https://github.com/THU-MIG/yolov10), [RTDETR](https://arxiv.org/abs/2304.08069), [SAM](https://github.com/facebookresearch/segment-anything), [CLIP](https://github.com/openai/CLIP), [DINOv2](https://github.com/facebookresearch/dinov2), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [BLIP](https://arxiv.org/abs/2201.12086), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [Depth-Anything](https://github.com/LiheYoung/Depth-Anything), [MODNet](https://github.com/ZHKKKe/MODNet) and others.

|                          Monocular Depth Estimation              |
| :--------------------------------------------------------------: |
| <img src='examples/depth-anything/demo.png'   width="800px"> |


|                        Panoptic Driving Perception                        |    Text-Detection-Recognition   |
| :----------------------------------------------------: | :------------------------------------------------: |
| <img src='examples/yolop/demo.png'  width="385px"> | <img src='examples/db/demo.png'  width="385x"> |

|                     Portrait Matting                     |
| :------------------------------------------------------: |
| <img src='examples/modnet/demo.png'   width="800px"> |


## Supported Models

|                               Model                               |                                                      Task / Type                                                      |           Example           | CUDA<br />f32 | CUDA<br />f16 |     TensorRT<br />f32     |     TensorRT<br />f16     |
| :---------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: | :--------------------------: | :-----------: | :-----------: | :------------------------: | :-----------------------: |
|           [YOLOv5](https://github.com/ultralytics/yolov5)           |                            Classification<br />Object Detection<br />Instance Segmentation                            |     [demo](examples/yolo)     |      ✅      |      ✅      |             ✅             |            ✅            |
|           [YOLOv6](https://github.com/meituan/YOLOv6)         |                           Object Detection                           |     [demo](examples/yolo)     |      ✅      |      ✅      |             ✅             |            ✅            |
|           [YOLOv7](https://github.com/WongKinYiu/yolov7)         |                            Object Detection                            |     [demo](examples/yolo)     |      ✅      |      ✅      |             ✅             |            ✅            |
|         [YOLOv8](https://github.com/ultralytics/ultralytics)         | Object Detection<br />Instance Segmentation<br />Classification<br />Oriented Object Detection<br />Keypoint Detection |     [demo](examples/yolo)     |      ✅      |      ✅      |             ✅             |            ✅            |
|            [YOLOv9](https://github.com/WongKinYiu/yolov9)            |                                                    Object Detection                                                    |     [demo](examples/yolo)     |      ✅      |      ✅      |             ✅             |            ✅            |
|            [YOLOv10](https://github.com/THU-MIG/yolov10)            |                                                    Object Detection                                                    |    [demo](examples/yolo)    |      ✅      |      ✅      |             ✅             |            ✅            |
|              [RTDETR](https://arxiv.org/abs/2304.08069)              |                                                    Object Detection                                                    |     [demo](examples/yolo)     |      ✅      |      ✅      |             ✅             |            ✅            |
|         [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)         |                                                 Instance Segmentation                                                 |    [demo](examples/yolo)    |      ✅      |      ✅      |             ✅             |            ✅            |
|        [YOLO-World](https://github.com/AILab-CVC/YOLO-World)        |                                                    Object Detection                                                    |   [demo](examples/yolo)   |      ✅      |      ✅      |             ✅             |            ✅            |
|         [DINOv2](https://github.com/facebookresearch/dinov2)         |                                                 Vision-Self-Supervised                                                 |     [demo](examples/dinov2)     |      ✅      |      ✅      |             ✅             |            ✅            |
|                [CLIP](https://github.com/openai/CLIP)                |                                                    Vision-Language                                                    |      [demo](examples/clip)      |      ✅      |      ✅      | ✅ visual<br />❌ textual | ✅ visual<br />❌ textual |
|              [BLIP](https://github.com/salesforce/BLIP)              |                                                    Vision-Language                                                    |      [demo](examples/blip)      |      ✅      |      ✅      | ✅ visual<br />❌ textual | ✅ visual<br />❌ textual |
|                [DB](https://arxiv.org/abs/1911.08947)                |                                                     Text Detection                                                     |       [demo](examples/db)       |      ✅      |      ✅      |             ✅             |            ✅            |
|               [SVTR](https://arxiv.org/abs/2205.00159)               |                                                    Text Recognition                                                    |      [demo](examples/svtr)      |      ✅      |      ✅      |             ✅             |            ✅            |
| [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) |                                                   Keypoint Detection                                                   |      [demo](examples/rtmo)      |      ✅      |      ✅      |             ❌             |            ❌            |
|             [YOLOPv2](https://arxiv.org/abs/2208.11434)             |                                              Panoptic Driving Perception                                              |     [demo](examples/yolop)     |      ✅      |      ✅      |             ✅             |            ✅            |
|    [Depth-Anything<br />(v1, v2)](https://github.com/LiheYoung/Depth-Anything)    |                                               Monocular Depth Estimation                                               | [demo](examples/depth-anything) |      ✅      |      ✅      |             ❌             |            ❌            |
|              [MODNet](https://github.com/ZHKKKe/MODNet)              |                                                     Image Matting                                                     |     [demo](examples/modnet)     |      ✅      |      ✅      |             ✅             |            ✅            |

## Installation

Refer to [ort docs](https://ort.pyke.io/setup/linking)

<details close>
<summary>For Linux or MacOS users</summary>

- Download from [ONNXRuntime Releases](https://github.com/microsoft/onnxruntime/releases)
- Then linking
  ```Shell
  export ORT_DYLIB_PATH=/Users/qweasd/Desktop/onnxruntime-osx-arm64-1.17.1/lib/libonnxruntime.1.17.1.dylib
  ```

</details>

## Quick Start

```Shell
cargo run -r --example yolo   # blip, clip, yolop, svtr, db, ...
```

## Integrate into your own project

Add `usls` as a dependency to your project's `Cargo.toml`

```Shell
cargo add usls
```

Or you can use specific commit

```Shell
usls = { git = "https://github.com/jamjamjon/usls", rev = "???sha???"}
```
