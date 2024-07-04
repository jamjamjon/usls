## Quick Start

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
