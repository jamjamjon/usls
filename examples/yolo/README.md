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

# Your customized YOLOv8 model
cargo run -r --example yolo -- --task detect --ver v8 --num-classes 6 --model xxx.onnx  # YOLOv8

# Classify
cargo run -r --example yolo -- --task classify --ver 5 --scale s --image-width 224 --image-height 224 --num-classes 1000 --use-imagenet-1k-classes # YOLOv5
cargo run -r --example yolo -- --task classify --ver 8 --scale n --image-width 224 --image-height 224 # YOLOv8 
cargo run -r --example yolo -- --task classify --ver 11 --scale n --image-width 224 --image-height 224  # YOLOv11 

# Detect
cargo run -r --example yolo -- --task detect --ver 5 --scale n --use-coco-80-classes  # YOLOv5 
cargo run -r --example yolo -- --task detect --ver 6 --scale n --use-coco-80-classes  # YOLOv6
cargo run -r --example yolo -- --task detect --ver 7 --scale t --use-coco-80-classes  # YOLOv7
cargo run -r --example yolo -- --task detect --ver 8 --scale n --use-coco-80-classes  # YOLOv8
cargo run -r --example yolo -- --task detect --ver 9 --scale t --use-coco-80-classes  # YOLOv9
cargo run -r --example yolo -- --task detect --ver 10 --scale n --use-coco-80-classes  # YOLOv10
cargo run -r --example yolo -- --task detect --ver 11 --scale n --use-coco-80-classes  # YOLOv11
cargo run -r --example yolo -- --task detect --ver 8 --model v8-s-world-v2-shoes.onnx  # YOLOv8-world

# Pose
cargo run -r --example yolo -- --task pose --ver 8 --scale n   # YOLOv8-Pose
cargo run -r --example yolo -- --task pose --ver 11 --scale n  # YOLOv11-Pose

# Segment
cargo run -r --example yolo -- --task segment --ver 5 --scale n  # YOLOv5-Segment
cargo run -r --example yolo -- --task segment --ver 8 --scale n  # YOLOv8-Segment
cargo run -r --example yolo -- --task segment --ver 11 --scale n  # YOLOv8-Segment

# Obb
cargo run -r --example yolo -- --ver 8 --task obb --scale n --image-width 1024 --image-height 1024 --source images/dota.png  # YOLOv8-Obb
cargo run -r --example yolo -- --ver 11 --task obb --scale n --image-width 1024 --image-height 1024 --source images/dota.png  # YOLOv11-Obb
```

**`cargo run -r --example yolo -- --help` for more options**

## Other YOLOv8 Solution Models

|          Model          |           Weights    |                                                                                                                                  
| :---------------------: | :------------------------------------------------------: | 
| Face-Landmark Detection |    [yolov8-n-face](https://github.com/jamjamjon/assets/releases/download/yolo/v8-n-face-fp16.onnx)                                                                                                                                                                                                                                                                              |
|     Head Detection     |         [yolov8-head](https://github.com/jamjamjon/assets/releases/download/yolo/v8-head-fp16.onnx)                                                                                                                                                                                                                                                                                  |
|     Fall Detection     |     [yolov8-falldown](https://github.com/jamjamjon/assets/releases/download/yolo/v8-falldown-fp16.onnx)                                                                                                                                                                                                                                                                             |
|     Trash Detection     |  [yolov8-plastic-bag](https://github.com/jamjamjon/assets/releases/download/yolo/v8-plastic-bag-fp16.onnx)                                                                                                                                                                                                                                                                             |
|       FaceParsing       | [yolov8-face-parsing-seg](https://github.com/jamjamjon/assets/releases/download/yolo/v8-face-parsing.onnx) | 
