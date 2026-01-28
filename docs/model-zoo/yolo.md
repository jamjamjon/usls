# YOLO Series

**usls** provides comprehensive support for the YOLO (You Only Look Once) family of models, spanning from YOLOv5 to the latest YOLO26.

## Supported Versions

| Model | Task / Description | Dynamic Batch | TensorRT | FP16 | Q8 | Q4f16 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| [YOLOv5](https://github.com/ultralytics/yolov5) | Classification, Detection, Segmentation | ✅ | ✅ | ✅ | ✅ | ❌ | 
| [YOLOv8](https://github.com/ultralytics/ultralytics) | Detection, Segmentation, Pose, OBB | ✅ | ✅ | ✅ | ✅ | ❌ | 
| [YOLO11](https://github.com/ultralytics/ultralytics) | Detection, Segmentation, Pose, OBB | ✅ | ✅ | ✅ | ✅ | ❌ | 
| [YOLOv12](https://github.com/sunsmarterjie/yolov12) | Detection, Segmentation, Classification | ✅ | ✅ | ✅ | ✅ | ✅ | 
| [YOLO26](https://github.com/ultralytics/ultralytics) | Detection, Segmentation, Pose, OBB | ✅ | ✅ | ✅ | ✅ | ✅ | 

## Quick Usage

```bash
# Run YOLOv11 object detection on CPU
cargo run -r --example yolo -- --task detect --ver 11 --scale n
```

## Performance Benchmarks

For detailed performance metrics across different hardware (CUDA, TensorRT, CPU), please refer to the [Home Page](../index.md#highlights).
