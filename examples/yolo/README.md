## Quick Start



## Options for all YOLOs

```Rust
// FastSAM
cargo run -r --example yolo -- --plot --task segment --version v8 --model FastSAM-s-dyn-f16.onnx

// YOLOv5-Classify
let options = options
    .with_yolo_version(args.version)
    .with_yolo_task(args.task)
    .with_model("../models/yolov5s-cls.onnx")?;

// YOLOv5-Detect
let options = options
    .with_yolo_version(YOLOVersion::V5)
    .with_yolo_task(YOLOTask::Detect)
    .with_model("../models/yolov5s.onnx")?;

// YOLOv5-Segment
let options = options
    .with_yolo_version(YOLOVersion::V5)
    .with_yolo_task(YOLOTask::Segment)
    .with_model("../models/yolov5s.onnx")?;

// YOLOv8-Detect
let options = options
    .with_yolo_version(YOLOVersion::V8)
    .with_yolo_task(YOLOTask::Detect)
    .with_model("yolov8m-dyn.onnx")?;

// YOLOv8-Classify
let options = options
    .with_yolo_version(YOLOVersion::V8)
    .with_yolo_task(YOLOTask::Classify)
    .with_model("yolov8m-cls-dyn.onnx")?;

// YOLOv8-Pose
let options = options
    .with_yolo_version(YOLOVersion::V8)
    .with_yolo_task(YOLOTask::Pose)
    .with_model("yolov8m-pose-dyn.onnx")?;

// YOLOv8-Segment
let options = options
    .with_yolo_version(YOLOVersion::V8)
    .with_yolo_task(YOLOTask::Segment)
    .with_model("yolov8m-seg-dyn.onnx")?;

// YOLOv8-Obb
let options = options
    .with_yolo_version(YOLOVersion::V8)
    .with_yolo_task(YOLOTask::Obb)
    .with_model("yolov8m-obb-dyn.onnx")?;

// YOLOv9-Detect
let options = options
    .with_yolo_version(YOLOVersion::V9)
    .with_yolo_task(YOLOTask::Detect)
    .with_model("yolov9-c-dyn-f16.onnx")?;

// YOLOv10-Detect
let options = options
    .with_yolo_version(YOLOVersion::V10)
    .with_yolo_task(YOLOTask::Detect)
    .with_model("yolov10n-dyn.onnx")?;
```



```shell
cargo run -r --example yolov8
```

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
