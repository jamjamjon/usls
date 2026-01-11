# Pose Estimation Examples

This directory contains examples for human pose estimation models.

## Examples

### RTMO
```bash
cargo run -F cuda --example pose-estimation -- rtmo --dtype f16 --device cuda:0 --processor-device cuda:0 
```

### RTMW
```bash
cargo run -F cuda --example pose-estimation -- rtmw --dtype f16 --device cuda:0 --processor-device cuda:0 
```

### RTMPose
```bash
cargo run -F cuda --example pose-estimation -- rtmpose --dtype f32 --device cuda:0 --processor-device cuda:0 
```

### DWPose
```bash
cargo run -F cuda --example pose-estimation -- dwpose --dtype f16 --device cuda:0 --processor-device cuda:0 
```
