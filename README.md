# usls

A Rust library integrated with **ONNXRuntime**, providing a collection of **Computer Vison** and **Vision-Language** models including [YOLOv8](https://github.com/ultralytics/ultralytics) `(Classification, Segmentation, Detection and Pose Detection)`, [YOLOv9](https://github.com/WongKinYiu/yolov9), [RTDETR](https://arxiv.org/abs/2304.08069), [CLIP](https://github.com/openai/CLIP), [DINOv2](https://github.com/facebookresearch/dinov2), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [BLIP](https://arxiv.org/abs/2201.12086), and others. Many execution providers are supported, sunch as `CUDA`, `TensorRT` and `CoreML`.


## Supported Models

|         Model         |         Example         |     CUDA(f32)     |     CUDA(f16)     |       TensorRT(f32)       |       TensorRT(f16)       | 
| :-------------------: | :----------------------: | :----------------: | :----------------: | :------------------------: | :-----------------------: | 
|   YOLOv8-detection   |   [demo](examples/yolov8)   |         ✅         |         ✅         |             ✅             |            ✅            |                    
|      YOLOv8-pose      |   [demo](examples/yolov8)   |         ✅         |         ✅         |             ✅             |            ✅            |   
| YOLOv8-classification |   [demo](examples/yolov8)   |         ✅         |         ✅         |             ✅             |            ✅            |               
|  YOLOv8-segmentation  |   [demo](examples/yolov8)   |         ✅         |         ✅         |             ✅             |            ✅            |               
|      YOLOv8-OBB      |    ***TODO***    | ***TODO*** | ***TODO*** |     ***TODO***     |    ***TODO***    |                                   |         
|        YOLOv9        |   [demo](examples/yolov9)   |         ✅         |         ✅         |             ✅             |            ✅            |                           
|        RT-DETR        |   [demo](examples/rtdetr)   |         ✅         |         ✅         |             ✅             |            ✅            |          
|        FastSAM        |  [demo](examples/fastsam)  |         ✅         |         ✅         |             ✅             |            ✅            |     
|      YOLO-World      | [demo](examples/yolo-world) |         ✅         |         ✅         |             ✅             |            ✅            |      
|        DINOv2        |   [demo](examples/dinov2)   |         ✅         |         ✅         |             ✅             |            ✅            |      
|         CLIP         |    [demo](examples/clip)    |         ✅         |         ✅         | ✅ visual<br />❌ textual | ✅ visual<br />❌ textual |                   
|         BLIP         |    [demo](examples/blip)    |         ✅         |         ✅         | ✅ visual<br />❌ textual | ✅ visual<br />❌ textual |     
|     OCR(DB, SVTR)     |    ***TODO***    | ***TODO*** | ***TODO*** |     ***TODO***     |    ***TODO***    |                                   |    

## Solution Models
Additionally, this repo also provides some solution models such as pedestrian `fall detection`, `head detection`, `trash detection`, and more.

|             Model             |             Example             |                                    Result                                    |
| :---------------------------: | :------------------------------: | :--------------------------------------------------------------------------: |
|    face-landmark detection    |    [demo](examples/yolov8-face)    |   <img src="./examples/yolov8-face/demo.jpg" width="400" height="300">  |
|        head detection        |    [demo](examples/yolov8-head)    |   <img src="./examples/yolov8-head/demo.jpg" width="400" height="300">   |
|      fall detection      |  [demo](examples/yolov8-falldown)  | <img src="./examples/yolov8-falldown/demo.jpg" width="400" height="300"> |
| trash detection | [demo](examples/yolov8-plastic-bag) |  <img src="./examples/yolov8-trash/demo.jpg" width="400" height="260">  |

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

#### 2. Add `usls` as a dependency to your project's `Cargo.toml:`

```
[dependencies]
usls = "0.0.1"
```

#### 3. Set model `Options` and build `model`, then you're ready to go.

```Rust
2use usls::{models::YOLO, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1.build model
    let options = Options::default()
        .with_model("../models/yolov8m-seg-dyn-f16.onnx")
        .with_trt(0) // using cuda(0) by default
	// when model with dynamic shapes
        .with_i00((1, 2, 4).into()) // dynamic batch
        .with_i02((416, 640, 800).into())   // dynamic height
        .with_i03((416, 640, 800).into())   // dynamic width
        .with_confs(&[0.4, 0.15]) // person: 0.4, others: 0.15
        .with_saveout("YOLOv8");    // save results
    let mut model = YOLO::new(&options)?;

    // 2.build dataloader
    let dl = DataLoader::default()
        .with_batch(model.batch.opt as usize)
        .load("./assets/")?;

    // 3.run
    for (xs, _paths) in dl {
        let _y = model.run(&xs)?;
    }
    Ok(())
}
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
