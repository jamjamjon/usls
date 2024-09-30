<h1 align='center'>YOLO-Series</h1>


|      Detection     |    Instance Segmentation   |   Pose     | 
| :---------------: | :------------------------: |:---------------: |
| <img src='https://github.com/jamjamjon/assets/releases/download/yolo/demo-det.png'  width="300px">  | <img src='https://github.com/jamjamjon/assets/releases/download/yolo/demo-seg.png'  width="300px"> |<img src='https://github.com/jamjamjon/assets/releases/download/yolo/demo-pose.png'  width="300px">  | 

|    Classification   |    Obb   |
| :------------------------: |:------------------------: |
|<img src='https://github.com/jamjamjon/assets/releases/download/yolo/demo-cls.png'  width="300px"> |<img src='https://github.com/jamjamjon/assets/releases/download/yolo/demo-obb-2.png'  width="628px">

|    Head Detection   |    Fall Detection   | Trash Detection   |
| :------------------------: |:------------------------: |:------------------------: |
|<img src='https://github.com/jamjamjon/assets/releases/download/yolo/demo-head.png'  width="300px"> |<img src='https://github.com/jamjamjon/assets/releases/download/yolo/demo-falldown.png'  width="300px">|<img src='https://github.com/jamjamjon/assets/releases/download/yolo/demo-trash.png'  width="300px">

|    YOLO-World   |    Face Parsing   | FastSAM   |
| :------------------------: |:------------------------: |:------------------------: |
|<img src='https://github.com/jamjamjon/assets/releases/download/yolo/demo-yolov8-world.png'  width="300px"> |<img src='https://github.com/jamjamjon/assets/releases/download/yolo/demo-face-parsing.png'  width="300px">|<img src='https://github.com/jamjamjon/assets/releases/download/yolo/demo-fastsam.png'  width="300px">





## Quick Start
```Shell

# customized
cargo run -r --example yolo -- --task detect --ver v8 --nc 6 --model xxx.onnx  # YOLOv8

# Classify
cargo run -r --example yolo -- --task classify --ver v5 --scale s --width 224 --height 224 --nc 1000  # YOLOv5
cargo run -r --example yolo -- --task classify --ver v8 --scale n --width 224 --height 224 --nc 1000  # YOLOv8 
cargo run -r --example yolo -- --task classify --ver v11 --scale n --width 224 --height 224 --nc 1000  # YOLOv11 

# Detect
cargo run -r --example yolo -- --task detect --ver v5 --scale n  # YOLOv5 
cargo run -r --example yolo -- --task detect --ver v6 --scale n  # YOLOv6
cargo run -r --example yolo -- --task detect --ver v7 --scale t  # YOLOv7
cargo run -r --example yolo -- --task detect --ver v8 --scale n  # YOLOv8
cargo run -r --example yolo -- --task detect --ver v9 --scale t  # YOLOv9
cargo run -r --example yolo -- --task detect --ver v10 --scale n  # YOLOv10
cargo run -r --example yolo -- --task detect --ver v11 --scale n  # YOLOv11
cargo run -r --example yolo -- --task detect --ver rtdetr --scale l  # RTDETR
cargo run -r --example yolo -- --task detect --ver v8 --nc 1 --model yolov8s-world-v2-shoes.onnx  # YOLOv8-world <local file>

# Pose
cargo run -r --example yolo -- --task pose --ver v8 --scale n   # YOLOv8-Pose
cargo run -r --example yolo -- --task pose --ver v11 --scale n  # YOLOv11-Pose

# Segment
cargo run -r --example yolo -- --task segment --ver v5 --scale n  # YOLOv5-Segment
cargo run -r --example yolo -- --task segment --ver v8 --scale n  # YOLOv8-Segment
cargo run -r --example yolo -- --task segment --ver v11 --scale n  # YOLOv8-Segment
cargo run -r --example yolo -- --task segment --ver v8 --model FastSAM-s-dyn-f16.onnx  # FastSAM <local file>

# Obb
cargo run -r --example yolo -- --ver v8 --task obb --scale n --width 1024 --height 1024 --source images/dota.png  # YOLOv8-Obb
cargo run -r --example yolo -- --ver v11 --task obb --scale n --width 1024 --height 1024 --source images/dota.png  # YOLOv11-Obb
```

**`cargo run -r --example yolo -- --help` for more options**


## YOLOs configs with `Options` 

<details open>
<summary>Use official YOLO Models</summary>

```Rust
let options = Options::default()
    .with_yolo_version(YOLOVersion::V5)  // YOLOVersion: V5, V6, V7, V8, V9, V10, RTDETR
    .with_yolo_task(YOLOTask::Classify)  // YOLOTask: Classify, Detect, Pose, Segment, Obb
    // .with_nc(80)
    // .with_names(&COCO_CLASS_NAMES_80)
    .with_model("xxxx.onnx")?;

```
</details>

<details open>
<summary>Cutomized your own YOLO model</summary>

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
</details>

## Other YOLOv8 Solution Models

|          Model          |    Weights   | Datasets|
|:---------------------: | :--------------------------: | :-------------------------------: |
| Face-Landmark Detection |   [yolov8-face-dyn-f16](https://github.com/jamjamjon/assets/releases/download/yolo/v8-n-face-dyn-f16.onnx)         | |
| Head Detection |   [yolov8-head-f16](https://github.com/jamjamjon/assets/releases/download/yolo/v8-head-f16.onnx)         | |
| Fall Detection |   [yolov8-falldown-f16](https://github.com/jamjamjon/assets/releases/download/yolo/v8-falldown-f16.onnx)          | |
| Trash Detection |   [yolov8-plastic-bag-f16](https://github.com/jamjamjon/assets/releases/download/yolo/v8-plastic-bag-f16.onnx)         | |
| FaceParsing |  [yolov8-face-parsing-dyn](https://github.com/jamjamjon/assets/releases/download/yolo/v8-face-parsing-dyn.onnx)  | [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing)<br />[[Processed YOLO labels]](https://github.com/jamjamjon/assets/releases/download/yolo/CelebAMask-HQ-YOLO-Labels.zip)[[Python Script]](../../scripts/CelebAMask-HQ-To-YOLO-Labels.py) |




## Export ONNX Models


<details close>
<summary>YOLOv5</summary>
    
[Here](https://docs.ultralytics.com/yolov5/tutorials/model_export/)

</details>


<details close>
<summary>YOLOv6</summary>

[Here](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX)

</details>


<details close>
<summary>YOLOv7</summary>

[Here](https://github.com/WongKinYiu/yolov7?tab=readme-ov-file#export)

</details>

<details close>
<summary>YOLOv8, YOLOv11</summary>
    
```Shell
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
</details>


<details close>
<summary>YOLOv9</summary>

[Here](https://github.com/WongKinYiu/yolov9/blob/main/export.py)

</details>

<details close>
<summary>YOLOv10</summary>

[Here](https://github.com/THU-MIG/yolov10#export)

</details>
