<h2 align="center">usls</h2>
<p align="center">
    <!-- Rust MSRV -->
    <a href='https://crates.io/crates/usls'>
        <img src='https://img.shields.io/crates/msrv/usls-yellow?' alt='Rust MSRV'>
    </a>
    <!-- ONNXRuntime MSRV -->
    <a href='https://github.com/microsoft/onnxruntime/releases'>
        <img src='https://img.shields.io/badge/onnxruntime-%3E%3D%201.19.0-3399FF' alt='ONNXRuntime MSRV'>
    </a>
    <!-- CUDA MSRV -->
    <a href='https://developer.nvidia.com/cuda-toolkit-archive'>
        <img src='https://img.shields.io/badge/CUDA-%3E%3D%2012.0-green' alt='CUDA MSRV'>
    </a>
     <!-- cuDNN MSRV -->
    <a href='https://developer.nvidia.com/cudnn-downloads'>
        <img src='https://img.shields.io/badge/cuDNN-%3E%3D%209.0-green4' alt='cuDNN MSRV'>
    </a>
    <!-- TensorRT MSRV -->
    <a href='https://developer.nvidia.com/tensorrt'>
        <img src='https://img.shields.io/badge/TensorRT-%3E%3D%2012.0-0ABF53' alt='TensorRT MSRV'>
    </a>
</p>
<p align="center">
    <!-- Examples Link -->
    <a href="./examples">
        <img src="https://img.shields.io/badge/Examples-1A86FD?&logo=anki" alt="Examples">
    </a>
    <!-- Docs.rs Link -->
    <a href='https://docs.rs/usls'>
        <img src='https://img.shields.io/badge/Docs-usls-yellow?&logo=docs.rs&color=FFA200' alt='Documentation'>
    </a>
</p>
<p align="center">
    <!-- CI Badge -->
    <a href="https://github.com/jamjamjon/usls/actions/workflows/rust-ci.yml">
        <img src="https://github.com/jamjamjon/usls/actions/workflows/rust-ci.yml/badge.svg" alt="Rust CI">
    </a>
    <a href='https://crates.io/crates/usls'>
        <img src='https://img.shields.io/crates/v/usls.svg' alt='Crates.io Version'>
    </a>
    <!-- Crates.io Downloads -->
    <a href="https://crates.io/crates/usls">
        <img alt="Crates.io Downloads" src="https://img.shields.io/crates/d/usls?&color=946CE6">
    </a>
</p>
<p align="center">
    <strong>⭐️ Star if helpful! ⭐️</strong>
</p>

**usls** is an evolving Rust library focused on inference for advanced **vision** and **vision-language** models, along with practical vision utilities.

- **SOTA Model Inference:** Supports a wide range of state-of-the-art vision and multi-modal models (typically with fewer than 1B parameters).
- **Multi-backend Acceleration:** Supports CPU, CUDA, TensorRT, and CoreML.
- **Easy Data Handling:** Easily read images, video streams, and folders with iterator support.
- **Rich Result Types:** Built-in containers for common vision outputs like bounding boxes (Hbb, Obb), polygons, masks, etc.
- **Annotation & Visualization:** Draw and display inference results directly, similar to OpenCV's `imshow()`.


## 🧩 Supported Models

- **YOLO Models**: [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv9](https://github.com/WongKinYiu/yolov9), [YOLOv10](https://github.com/THU-MIG/yolov10), [YOLO11](https://github.com/ultralytics/ultralytics), [YOLOv12](https://github.com/sunsmarterjie/yolov12)
- **SAM Models**: [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/segment-anything-2), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [EdgeSAM](https://github.com/chongzhou96/EdgeSAM), [SAM-HQ](https://github.com/SysCV/sam-hq), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
- **Vision Models**: [RT-DETR](https://arxiv.org/abs/2304.08069), [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo), [Depth-Anything](https://github.com/LiheYoung/Depth-Anything), [DINOv2](https://github.com/facebookresearch/dinov2), [MODNet](https://github.com/ZHKKKe/MODNet), [Sapiens](https://arxiv.org/abs/2408.12569), [DepthPro](https://github.com/apple/ml-depth-pro), [FastViT](https://github.com/apple/ml-fastvit), [BEiT](https://github.com/microsoft/unilm/tree/master/beit), [MobileOne](https://github.com/apple/ml-mobileone)
- **Vision-Language Models**: [CLIP](https://github.com/openai/CLIP), [jina-clip-v1](https://huggingface.co/jinaai/jina-clip-v1), [BLIP](https://arxiv.org/abs/2201.12086), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [Florence2](https://arxiv.org/abs/2311.06242), [Moondream2](https://github.com/vikhyat/moondream/tree/main)
- **OCR-Related Models**: [FAST](https://github.com/czczup/FAST), [DB(PaddleOCR-Det)](https://arxiv.org/abs/1911.08947), [SVTR(PaddleOCR-Rec)](https://arxiv.org/abs/2205.00159), [SLANet](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/table_recognition/algorithm_table_slanet.html), [TrOCR](https://huggingface.co/microsoft/trocr-base-printed), [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

<details>
<summary>Full list of supported models (click to expand)</summary>

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
| [RF-DETR](https://github.com/roboflow/rf-detr)                                                                    | Object Detection                                                                                                             | [demo](examples/rfdetr)         | ✅     | ✅             | ✅             |                    |                    |
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



## 🛠️ Installation
**Note:** It is recommended to use the GitHub repository as the source, since the crates.io version may not be up-to-date.

```toml
[dependencies]
usls = { git = "https://github.com/jamjamjon/usls" }

# crates.io version
usls = "latest-version"
```

## ⚡ Cargo Features
- **ONNXRuntime-related features (enabled by default)**, provide model inference and model zoo support:
    - **`ort-download-binaries`**  (**default**): Automatically downloads prebuilt `ONNXRuntime` binaries for supported platforms. Provides core model loading and inference capabilities using the `CPU` execution provider.
    - **`ort-load-dynamic `** Dynamic linking. You'll need to compile `ONNXRuntime` from [source](https://github.com/microsoft/onnxruntime) or download a [precompiled package](https://github.com/microsoft/onnxruntime/releases), and then link it manually. [See the guide here](https://ort.pyke.io/setup/linking#dynamic-linking).
    
    - **`cuda`**: Enables the NVIDIA `CUDA` provider. Requires `CUDA` toolkit and `cuDNN` installed.
    - **`trt`**: Enables the NVIDIA `TensorRT` provider. Requires `TensorRT` libraries installed.
    - **`mps`**: Enables the Apple `CoreML` provider for macOS.

- **If you only need basic features** (such as image/video reading, result visualization, etc.), you can disable the default features to minimize dependencies:
    ```shell
    usls = { git = "https://github.com/jamjamjon/usls", default-features = false }
    ```
    - **`video`** : Enable video stream reading, and video writing.(Note: Powered by [video-rs](https://github.com/oddity-ai/video-rs) and [minifb](https://github.com/emoon/rust_minifb). Check their repositories for potential issues.)

## ✨ Example

- Model Inference
    ```shell
    cargo run -r --example yolo   # CPU
    cargo run -r -F cuda --example yolo -- --device cuda:0  # GPU
    ```

- Reading Images
    ```rust
    // Read a single image
    let image = DataLoader::try_read_one("./assets/bus.jpg")?;

    // Read multiple images
    let images = DataLoader::try_read_n(&["./assets/bus.jpg", "./assets/cat.png"])?;

    // Read all images in a folder
    let images = DataLoader::try_read_folder("./assets")?;

    // Read images matching a pattern (glob)
    let images = DataLoader::try_read_pattern("./assets/*.Jpg")?;

    // Load images and iterate
    let dl = DataLoader::new("./assets")?.with_batch(2).build()?;
    for images in dl.iter() {
        // Code here
    }
    ```

- Reading Video
    ```rust
    let dl = DataLoader::new("http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4")?
        .with_batch(1)
        .with_nf_skip(2)
        .with_progress_bar(true)
        .build()?;
    for images in dl.iter() {
        // Code here
    }
    ```

- Annotate
    ```rust
    let annotator = Annotator::default();
    let image = DataLoader::try_read_one("./assets/bus.jpg")?;
    // hbb
    let hbb = Hbb::default()
            .with_xyxy(669.5233, 395.4491, 809.0367, 878.81226)
            .with_id(0)
            .with_name("person")
            .with_confidence(0.87094545);
    let _ = annotator.annotate(&image, &hbb)?;

    // keypoints
    let keypoints: Vec<Keypoint> = vec![
        Keypoint::default()
            .with_xy(139.35767, 443.43655)
            .with_id(0)
            .with_name("nose")
            .with_confidence(0.9739332),
        Keypoint::default()
            .with_xy(147.38545, 434.34055)
            .with_id(1)
            .with_name("left_eye")
            .with_confidence(0.9098319),
        Keypoint::default()
            .with_xy(128.5701, 434.07516)
            .with_id(2)
            .with_name("right_eye")
            .with_confidence(0.9320564),
    ];
    let _ = annotator.annotate(&image, &keypoints)?;
    ```


- Visualizing Inference Results and Exporting Video
    ```rust
    let dl = DataLoader::new(args.source.as_str())?.build()?;
    let mut viewer = Viewer::default().with_window_scale(0.5);

    for images in &dl {
        // Check if the window exists and is open
        if viewer.is_window_exist() && !viewer.is_window_open() {
            break;
        }

        // Show image in window
        viewer.imshow(&images[0])?;

        // Handle key events and delay
        if let Some(key) = viewer.wait_key(1) {
            if key == usls::Key::Escape {
                break;
            }
        }

        // Your custom code here

        // Write video frame (requires video feature)
        // if args.save_video {
        //     viewer.write_video_frame(&images[0])?;
        // }
    }
    ```

**All examples are located in the [examples](./examples/) directory.**

## ❓ FAQ
See issues or open a new discussion.

## 🤝 Contributing

Contributions are welcome! If you have suggestions, bug reports, or want to add new features or models, feel free to open an issue or submit a pull request.  


## 📜 License

This project is licensed under [LICENSE](LICENSE).
