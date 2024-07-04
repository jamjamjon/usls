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
| Instance Segmentation | `<img src='examples/yolov8/demo-seg.png' width="800px">`  |
|    Classification    | `<img src='examples/yolov8/demo-cls.png' width="800px">`  |
|       Detection       | `<img src='examples/yolov8/demo-det.png' width="800px">`  |
|         Pose         | `<img src='examples/yolov8/demo-pose.png' width="800px">` |

## Other YOLOv8 Solution Models

|          Model          |                                                                                                       Weights                                                                                                       |                                     Result                                     | Datasets                                                                                                                                                                                                                                                                                                                  |
| :---------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Face-Landmark Detection |                                                      [yolov8-face-dyn-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/yolov8-face-dyn-f16.onnx)                                                      |   `<img src='examples/yolov8-face/demo.png'  width="220px" height="180px">`   |                                                                                                                                                                                                                                                                                                                           |
|     Head Detection     |                                                          [yolov8-head-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/yolov8-head-f16.onnx)                                                          |   `<img src='examples/yolov8-head/demo.png'  width="220px" height="180px">`   |                                                                                                                                                                                                                                                                                                                           |
|     Fall Detection     |                                                      [yolov8-falldown-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/yolov8-falldown-f16.onnx)                                                      | `<img src='examples/yolov8-falldown/demo.png'  width="220px" height="180px">` |                                                                                                                                                                                                                                                                                                                           |
|     Trash Detection     |                                                   [yolov8-plastic-bag-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/yolov8-plastic-bag-f16.onnx)                                                   |  `<img src='examples/yolov8-trash/demo.png'  width="250px" height="180px">`  |                                                                                                                                                                                                                                                                                                                           |
|         FastSAM         |                                                        [FastSAM-s-dyn-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/FastSAM-s-dyn-f16.onnx)                                                        |                                                                                |                                                                                                                                                                                                                                                                                                                           |
|       FaceParsing       | [face-parsing-dyn](https://github.com/jamjamjon/assets/releases/download/v0.0.1/face-parsing-dyn.onnx)<br />[face-parsing-dyn-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/face-parsing-dyn-f16.onnx) |                                                                                | [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing)<br />[[Processed YOLO labels]](https://github.com/jamjamjon/assets/releases/download/v0.0.1/CelebAMask-HQ-YOLO-Labels.zip)[[Python Script]](https://github.com/jamjamjon/assets/releases/download/v0.0.1/CelebAMask-HQ-YOLO-Labels.zip) |
|                        |                                                                                                                                                                                                                      |                                                                                |                                                                                                                                                                                                                                                                                                                           |
|                        |                                                                                                                                                                                                                      |                                                                                |                                                                                                                                                                                                                                                                                                                           |
|                        |                                                                                                                                                                                                                      |                                                                                |                                                                                                                                                                                                                                                                                                                           |
