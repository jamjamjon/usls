# Detection Examples

## Quick Start

### rfdetr
```
# Model and processor on CUDA
cargo run -F cuda-full --example object-detection -- rfdetr --dtype fp16 --device cuda:0 --processor-device cuda:0

# TensorRT for model, CUDA for image processor
cargo run -F tensorrt-full --example object-detection -- rfdetr --dtype fp32 --device tensorrt:0 --processor-device cuda:0
```

### dfine
```
# Model and processor on CUDA
cargo run -F cuda-full --example object-detection -- dfine --dtype fp32 --device cuda:0 --processor-device cuda:0
```

### deim
```
# Model and processor on CUDA
cargo run -F cuda-full --example object-detection --deim --dtype fp32 --device cuda:0 --processor-device cuda:0 --ver 1
```

### deimv2
```
# Model and processor on CUDA
cargo run -F cuda-full --example object-detection -- deim --dtype fp32 --device cuda:0 --processor-device cuda:0 --ver 2
```

### rt-detr
```
# Model and processor on CUDA
cargo run -F cuda-full --example object-detection -- rtdetr --dtype fp32 --device cuda:0 --processor-device cuda:0
```

