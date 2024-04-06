# usls

A Rust library integrated with **ONNXRuntime**, providing a collection of **Computer Vison** and **Vision-Language** models including [YOLOv8](https://github.com/ultralytics/ultralytics) `(Classification, Segmentation, Detection and Pose Detection)`, [YOLOv9](https://github.com/WongKinYiu/yolov9), [RTDETR](https://arxiv.org/abs/2304.08069), [CLIP](https://github.com/openai/CLIP), [DINOv2](https://github.com/facebookresearch/dinov2), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [BLIP](https://arxiv.org/abs/2201.12086), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and others. Many execution providers are supported, sunch as `CUDA`, `TensorRT` and `CoreML`.

## Supported Models

|                               Model                               |         Example         | CUDA<br />f32 | CUDA<br />f16 |     TensorRT<br />f32     |     TensorRT<br />f16     |
| :---------------------------------------------------------------: | :----------------------: | :-----------: | :-----------: | :------------------------: | :-----------------------: |
|                    **YOLOv8-detection**                    |   [demo](examples/yolov8)   |      ✅      |      ✅      |             ✅             |            ✅            |
|                       **YOLOv8-pose**                       |   [demo](examples/yolov8)   |      ✅      |      ✅      |             ✅             |            ✅            |
|                  **YOLOv8-classification**                  |   [demo](examples/yolov8)   |      ✅      |      ✅      |             ✅             |            ✅            |
|                   **YOLOv8-segmentation**                   |   [demo](examples/yolov8)   |      ✅      |      ✅      |             ✅             |            ✅            |
|                       **YOLOv8-OBB**                       |           TODO           |     TODO     |     TODO     |            TODO            |           TODO           |
|                         **YOLOv9**                         |   [demo](examples/yolov9)   |      ✅      |      ✅      |             ✅             |            ✅            |
|                         **RT-DETR**                         |   [demo](examples/rtdetr)   |      ✅      |      ✅      |             ✅             |            ✅            |
|                         **FastSAM**                         |  [demo](examples/fastsam)  |      ✅      |      ✅      |             ✅             |            ✅            |
|                       **YOLO-World**                       | [demo](examples/yolo-world) |      ✅      |      ✅      |             ✅             |            ✅            |
|                         **DINOv2**                         |   [demo](examples/dinov2)   |      ✅      |      ✅      |             ✅             |            ✅            |
|                          **CLIP**                          |    [demo](examples/clip)    |      ✅      |      ✅      | ✅ visual<br />❌ textual | ✅ visual<br />❌ textual |
|                          **BLIP**                          |    [demo](examples/blip)    |      ✅      |      ✅      | ✅ visual<br />❌ textual | ✅ visual<br />❌ textual |
|   [**DB(Text Detection)**](https://arxiv.org/abs/1911.08947)   |     [demo](examples/db)     |      ✅      |      ❌      |             ✅             |            ✅            |
| [**SVTR(Text Recognition)**](https://arxiv.org/abs/2205.00159) |    [demo](examples/svtr)    |      ✅      |      ❌      |             ✅             |            ✅            |

## Solution Models

Additionally, this repo also provides some solution models such as pedestrian `fall detection`, `head detection`, `trash detection`, and more.

|                                       Model                                       |             Example             |
| :--------------------------------------------------------------------------------: | :------------------------------: |
|    **text detection<br />(PPOCR-det v3, v4)**<br />**通用文本检测**    |         [demo](examples/db)         |
| **text recognition<br />(PPOCR-rec v3, v4)**<br />**中英文-文本识别** |        [demo](examples/svtr)        |
|         **face-landmark detection**<br />**人脸 & 关键点检测**         |    [demo](examples/yolov8-face)    |
|                 **head detection**<br />  **人头检测**                 |    [demo](examples/yolov8-head)    |
|                 **fall detection**<br />  **摔倒检测**                 |  [demo](examples/yolov8-falldown)  |
|                **trash detection**<br />  **垃圾检测**                | [demo](examples/yolov8-plastic-bag) |

## Demo

```
cargo run -r --example yolov8   # fastsam, yolov9, blip, clip, dinov2, yolo-world...
```

## Integrate into your own project

#### 1. Install [ort](https://github.com/pykeio/ort)

check **[ort guide](https://ort.pyke.io/setup/linking)**

<details close>
<summary>For Linux or MacOS users</summary>

- Firstly, download from latest release from [ONNXRuntime Releases](https://github.com/microsoft/onnxruntime/releases)
- Then linking
  ```shell
  export ORT_DYLIB_PATH=/Users/qweasd/Desktop/onnxruntime-osx-arm64-1.17.1/lib/libonnxruntime.1.17.1.dylib
  ```

</details>

#### 2. Add `usls` as a dependency to your project's `Cargo.toml`

```shell
cargo add --git https://github.com/jamjamjon/usls
```

#### 3. Set `Options` and build model

```Rust
let options = Options::default()
    .with_model("../models/yolov8m-seg-dyn-f16.onnx");
let mut model = YOLO::new(&options)?;
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
        .with_confs(&[0.4, 0.15]) // person: 0.4, others: 0.15
    ```

- Go check [Options](src/options.rs) for more model options.



#### 4. Prepare inputs, and then you're ready to go

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

#### 5. Annotate and save results
```Rust
let annotator = Annotator::default().with_saveout("YOLOv8");
annotator.annotate(&x, &y);
```


## Script: converte ONNX model from `float32` to `float16`

```python
import onnx
from pathlib import Path
from onnxconverter_common import float16

model_f32 = "onnx_model.onnx"
model_f16 = float16.convert_float_to_float16(onnx.load(model_f32))
saveout = Path(model_f32).with_name(Path(model_f32).stem + "-f16.onnx")
onnx.save(model_f16, saveout)
```
