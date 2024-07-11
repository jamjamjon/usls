
## TODO: Demo pictures(combine into one)



### Quick Start for all YOLOs
```Shell

# Classify
cargo run -r --example yolo -- --task classify --version v5  # YOLOv5 
cargo run -r --example yolo -- --task classify --version v8  # YOLOv8 

# Detect
cargo run -r --example yolo -- --task detect --version v5  # YOLOv5 
cargo run -r --example yolo -- --task detect --version v6  # YOLOv6
cargo run -r --example yolo -- --task detect --version v7  # YOLOv7
cargo run -r --example yolo -- --task detect --version v8  # YOLOv8
cargo run -r --example yolo -- --task detect --version v9  # YOLOv9
cargo run -r --example yolo -- --task detect --version v10 # YOLOv10
cargo run -r --example yolo -- --task detect --version rtdetr  # YOLOv8-RTDETR
cargo run -r --example yolo -- --task detect --version v8 --model yolov8s-world-v2-shoes.onnx  # YOLOv8-world

# Pose
cargo run -r --example yolo -- --task pose --version v8  # YOLOv8-Pose

# Segment
cargo run -r --example yolo -- --task segment --version v5  # YOLOv5-Segment
cargo run -r --example yolo -- --task segment --version v8  # YOLOv8-Segment
cargo run -r --example yolo -- --task segment --version v8 --model FastSAM-s-dyn-f16.onnx  # FastSAM

# Obb
cargo run -r --example yolo -- --task obb --version v8  # YOLOv8-Obb
```

**Some other options**  
`--source` to specify the input image  
`--model` to specify the ONNX model  
`--width --height` to specify the resolution  
`--nc` to specify the number of model's classes  
`--plot` to annotate   
`--profile` to profile  
`--cuda --trt --coreml` to select device  
`--device_id` to decide which device to use  
`--half` to use float16 when using TensorRT EP  



### YOLOs configs with `Options` 

**Use `YOLOVersion` and `YOLOTask`**
```Rust
let options = Options::default()
    .with_yolo_version(YOLOVersion::V5)  // YOLOVersion: V5, V6, V7, V8, V9, V10, RTDETR
    .with_yolo_task(YOLOTask::Classify)  // YOLOTask: Classify, Detect, Pose, Segment, Obb
    .with_model("xxxx.onnx")?;

```

**Cutomized your YOLOs model**
```Rust
// This config is for YOLOv8-Segment 
use usls::{AnchorsPosition, BoxType, ClssType, YOLOPreds};

let options = Options::default()
    .with_yolo_preds(
        YOLOPreds {
            bbox: Some(BoxType::Cxcywh),
            clss: ClssType::Clss,
            coefs: Some(true),
            anchors: Some(AnchorsPosition::After),
            ..Default::default()
        }
    )
    .with_model("xxxx.onnx")?;
```

# -----------------------------------------------------------

## Export `YOLOv8` ONNX Models

```bash
pip install -U ultralytics

# export onnx model with dynamic shapes
yolo export model=yolov8m.pt format=onnx simplify dynamic
yolo export model=yolov8m-cls.pt format=onnx simplify dynamic
yolo export model=yolov8m-pose.pt format=onnx simplify dynamic
yolo export model=yolov8m-seg.pt format=onnx simplify dynamic
yolo export model=yolov8m-obb.pt format=onnx simplify dynamic

# export onnx model with fixed shapes
yolo export model=yolov8m.pt format=onnx simplify
yolo export model=yolov8m-cls.pt format=onnx simplify
yolo export model=yolov8m-pose.pt format=onnx simplify
yolo export model=yolov8m-seg.pt format=onnx simplify
yolo export model=yolov8m-obb.pt format=onnx simplify
```

## Result

|         Task         | Annotated image                                             |
| :-------------------: | ----------------------------------------------------------- |
|          Obb          |                                                             |
| Instance Segmentation | <img src='examples/yolov8/demos/seg.png' height="300px">  |
|    Classification    | <img src='examples/yolov8/demos/cls.png' height="300px">  |
|       Detection       | <img src='examples/yolov8/demos/det.png' height="300px">  |
|         Pose         | <img src='examples/yolov8/demos/pose.png' height="300px"> |

## Other YOLOv8 Solution Models

|          Model          |    Weights                | Result              | Datasets                                                                                                                                                                                                                                                                                                                  |
| :---------------------: | :--------------------------: | :-------------------------------: | ------ |
| Face-Landmark Detection |                                                      [yolov8-face-dyn-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/yolov8-face-dyn-f16.onnx)                                      | <img src='examples/yolov8/demos/face.png' height="300px">  |                                                                                                                                                                                                                                                                                                                           |
|     Head Detection     |                                                          [yolov8-head-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/yolov8-head-f16.onnx)                                                          | <img src='examples/yolov8/demos/head.png' height="300px"> |                                                                                                                                                                                                                                                                                                                           |
|     Fall Detection     |                                                      [yolov8-falldown-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/yolov8-falldown-f16.onnx)                                                      |<img src='examples/yolov8/demos/falldown.png' height="300px"> |                                                                                                                                                                                                                                                                                                                           |
|     Trash Detection     |                                                   [yolov8-plastic-bag-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/yolov8-plastic-bag-f16.onnx)                                                   |  <img src='examples/yolov8/demos/trash.png' height="300px">  |                                                                                                                                                                                                                                                                                                                           |
|         FastSAM         |                                                        [FastSAM-s-dyn-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/FastSAM-s-dyn-f16.onnx)                                                        |                                                                                |         <img src='examples/yolov8/demos/fastsam.png' height="300px">                        |
|       FaceParsing       | [face-parsing-dyn](https://github.com/jamjamjon/assets/releases/download/v0.0.1/face-parsing-dyn.onnx) |        <img src='examples/yolov8/demos/face-parsing.png' height="300px">                                                                        | [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing)<br />[[Processed YOLO labels]](https://github.com/jamjamjon/assets/releases/download/v0.0.1/CelebAMask-HQ-YOLO-Labels.zip)[[Python Script]](https://github.com/jamjamjon/assets/releases/download/v0.0.1/CelebAMask-HQ-YOLO-Labels.zip) |
