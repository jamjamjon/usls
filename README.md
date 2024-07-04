# usls

[![Static Badge](https://img.shields.io/crates/v/usls.svg?style=for-the-badge&logo=rust)](https://crates.io/crates/usls) [![Static Badge](https://img.shields.io/badge/Documents-usls-blue?style=for-the-badge&logo=docs.rs)](https://docs.rs/usls) [![Static Badge](https://img.shields.io/badge/GitHub-black?style=for-the-badge&logo=github)](https://github.com/jamjamjon/usls) ![Static Badge](https://img.shields.io/crates/d/usls?style=for-the-badge)

A Rust library integrated with **ONNXRuntime**, providing a collection of **Computer Vison** and **Vision-Language** models including [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv9](https://github.com/WongKinYiu/yolov9), [YOLOv10](https://github.com/THU-MIG/yolov10), [RTDETR](https://arxiv.org/abs/2304.08069), [CLIP](https://github.com/openai/CLIP), [DINOv2](https://github.com/facebookresearch/dinov2), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [BLIP](https://arxiv.org/abs/2201.12086), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [Depth-Anything](https://github.com/LiheYoung/Depth-Anything), [MODNet](https://github.com/ZHKKKe/MODNet) and others.

|                          Depth-Anything                          |
| :--------------------------------------------------------------: |
| `<img src='examples/depth-anything/demo.png'   width="800px">` |

|                        YOLOP-v2                        |    Text-Detection-Recognition<br />PPOCRv3, v4    |
| :----------------------------------------------------: | :------------------------------------------------: |
| `<img src='examples/yolop/demo.png'  width="385px">` | `<img src='examples/db/demo.png'  width="385x">` |

|                     Portrait Matting                     |
| :------------------------------------------------------: |
| `<img src='examples/modnet/demo.png'   width="800px">` |

|                           YOLOv8-Obb                           |
| :------------------------------------------------------------: |
| `<img src='examples/yolov8/demo-obb-2.png'   width="800px">` |

## Supported Models

|                               Model                               |                                                      Task / Type                                                      |           Example           | CUDA<br />f32 | CUDA<br />f16 |     TensorRT<br />f32     |     TensorRT<br />f16     |
| :---------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: | :--------------------------: | :-----------: | :-----------: | :------------------------: | :-----------------------: |
|           [YOLOv5](https://github.com/ultralytics/yolov5)           |                            Classification<br />Object Detection<br />Instance Segmentation                            |     [demo](examples/yolov5)     |      ✅      |      ✅      |             ✅             |            ✅            |
|         [YOLOv8](https://github.com/ultralytics/ultralytics)         | Object Detection<br />Instance Segmentation<br />Classification<br />Oriented Object Detection<br />Keypoint Detection |     [demo](examples/yolov8)     |      ✅      |      ✅      |             ✅             |            ✅            |
|            [YOLOv9](https://github.com/WongKinYiu/yolov9)            |                                                    Object Detection                                                    |     [demo](examples/yolov9)     |      ✅      |      ✅      |             ✅             |            ✅            |
|            [YOLOv10](https://github.com/THU-MIG/yolov10)            |                                                    Object Detection                                                    |    [demo](examples/yolov10)    |      ✅      |      ✅      |             ✅             |            ✅            |
|              [RTDETR](https://arxiv.org/abs/2304.08069)              |                                                    Object Detection                                                    |     [demo](examples/rtdetr)     |      ✅      |      ✅      |             ✅             |            ✅            |
|         [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)         |                                                 Instance Segmentation                                                 |    [demo](examples/fastsam)    |      ✅      |      ✅      |             ✅             |            ✅            |
|        [YOLO-World](https://github.com/AILab-CVC/YOLO-World)        |                                                    Object Detection                                                    |   [demo](examples/yolo-world)   |      ✅      |      ✅      |             ✅             |            ✅            |
|         [DINOv2](https://github.com/facebookresearch/dinov2)         |                                                 Vision-Self-Supervised                                                 |     [demo](examples/dinov2)     |      ✅      |      ✅      |             ✅             |            ✅            |
|                [CLIP](https://github.com/openai/CLIP)                |                                                    Vision-Language                                                    |      [demo](examples/clip)      |      ✅      |      ✅      | ✅ visual<br />❌ textual | ✅ visual<br />❌ textual |
|              [BLIP](https://github.com/salesforce/BLIP)              |                                                    Vision-Language                                                    |      [demo](examples/blip)      |      ✅      |      ✅      | ✅ visual<br />❌ textual | ✅ visual<br />❌ textual |
|                [DB](https://arxiv.org/abs/1911.08947)                |                                                     Text Detection                                                     |       [demo](examples/db)       |      ✅      |      ✅      |             ✅             |            ✅            |
|               [SVTR](https://arxiv.org/abs/2205.00159)               |                                                    Text Recognition                                                    |      [demo](examples/svtr)      |      ✅      |      ✅      |             ✅             |            ✅            |
| [RTMO](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) |                                                   Keypoint Detection                                                   |      [demo](examples/rtmo)      |      ✅      |      ✅      |             ❌             |            ❌            |
|             [YOLOPv2](https://arxiv.org/abs/2208.11434)             |                                              Panoptic Driving Perception                                              |     [demo](examples/yolop)     |      ✅      |      ✅      |             ✅             |            ✅            |
|    [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)    |                                               Monocular Depth Estimation                                               | [demo](examples/depth-anything) |      ✅      |      ✅      |             ❌             |            ❌            |
|              [MODNet](https://github.com/ZHKKKe/MODNet)              |                                                     Image Matting                                                     |     [demo](examples/modnet)     |      ✅      |      ✅      |             ✅             |            ✅            |

## Installation

Refer to **[ort guide](https://ort.pyke.io/setup/linking)**

<details close>
<summary>For Linux or MacOS users</summary>

- Firstly, download from latest release from [ONNXRuntime Releases](https://github.com/microsoft/onnxruntime/releases)
- Then linking
  ```Shell
  export ORT_DYLIB_PATH=/Users/qweasd/Desktop/onnxruntime-osx-arm64-1.17.1/lib/libonnxruntime.1.17.1.dylib
  ```

</details>

## Demo

```Shell
cargo run -r --example yolov8   # yolov10, blip, clip, yolop, svtr, db, yolo-world, ...
```

## Integrate into your own project

<details close>
<summary>Expand</summary>

### 1. Add `usls` as a dependency to your project's `Cargo.toml`

```Shell
cargo add usls
```

Or you can use specific commit

```Shell
usls = { git = "https://github.com/jamjamjon/usls", rev = "???sha???"}
```

### 2. Set `Options` and build model

```Rust
let options = Options::default()
    .with_model("../models/yolov8m-seg-dyn-f16.onnx");
let mut model = YOLO::new(options)?;
```

- If you want to run your model with TensorRT or CoreML

  ```Rust
  let options = Options::default()
      .with_trt(0) // using cuda by default
      // .with_coreml(0) 
  ```
- If your model has dynamic shapes

  ```Rust
  let options = Options::default()
      .with_i00((1, 2, 4).into()) // dynamic batch
      .with_i02((416, 640, 800).into())   // dynamic height
      .with_i03((416, 640, 800).into())   // dynamic width
  ```
- If you want to set a confidence level for each category

  ```Rust
  let options = Options::default()
      .with_confs(&[0.4, 0.15]) // class 0: 0.4, others: 0.15
  ```
- Go check [Options](src/options.rs) for more model options.

#### 3. Prepare inputs, and then you're ready to go

- Build `DataLoader` to load images

```Rust
let dl = DataLoader::default()
    .with_batch(model.batch.opt as usize)
    .load("./assets/")?;

for (xs, _paths) in dl {
    let _y = model.run(&xs)?;
}
```

- Or simply read one image

```Rust
let x = vec![DataLoader::try_read("./assets/bus.jpg")?];
let y = model.run(&x)?;
```

#### 4. Annotate and save results

```Rust
let annotator = Annotator::default().with_saveout("YOLOv8");
annotator.annotate(&x, &y);
```

#### 5. Get results

The inference outputs of provided models will be saved to `Vec<Y>`.

```Rust
pub struct Y {
    probs: Option<Prob>,
    bboxes: Option<Vec<Bbox>>,
    keypoints: Option<Vec<Vec<Keypoint>>>,
    mbrs: Option<Vec<Mbr>>,
    polygons: Option<Vec<Polygon>>,
    texts: Option<Vec<String>>,
    masks: Option<Vec<Mask>>,
    embedding: Option<Embedding>,
}
```

- You can get detection bboxes with `y.bboxes()`:

  ```Rust
  let ys = model.run(&xs)?;
  for y in ys {
      // bboxes
      if let Some(bboxes) = y.bboxes() {
          for bbox in bboxes {
              println!(
                  "Bbox: {}, {}, {}, {}, {}, {}",
                  bbox.xmin(),
                  bbox.ymin(),
                  bbox.xmax(),
                  bbox.ymax(),
                  bbox.confidence(),
                  bbox.id(),
              )
          }
      }
  }
  ```

  More `Bbox` methods here: `src/ys/bbox.rs`
- Other tasks results can be found at: `src/ys/y.rs`

</details>
