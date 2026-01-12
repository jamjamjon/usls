# Multi-Object Tracking (MOT) Examples

This directory contains examples for multi-object tracking using different detection models with ByteTrack.

## Example

### Use YOLO As Detector
```bash
cargo run -r --features mot,viewer,video,cuda-full --example bytetrack -- --source ../7.mp4 --retain-classes 0 yolo-track --device cuda --processor-device cuda --scale n --ver 8 # person: 0
```

### Use RFDETR As Detector
```bash
cargo run -r --features mot,viewer,video,cuda-full --example bytetrack -- --source ../7.mp4 --retain-classes 1 rfdetr-track --device cuda --processor-device cuda --scale n # person: 1
```
