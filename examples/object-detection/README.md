# Detection Examples

**TODO: Simple description for this example**


## Quick Start

### rfdetr
```
cargo run -F cuda --example object-detection rfdetr --dtype fp16 --device cuda:0 --processor-device cuda:0
```

### dfine
```
cargo run -F cuda --example object-detection dfine --dtype fp32 --device cuda:0 --processor-device cuda:0
```

### deim
```
cargo run -F cuda --example object-detection deim --dtype fp32 --device cuda:0 --processor-device cuda:0 --ver 1
```

### deimv2
```
cargo run -F cuda --example object-detection deim --dtype fp32 --device cuda:0 --processor-device cuda:0 --ver 2
```

### rt-detr
```
cargo run -F cuda --example object-detection rt-detr --dtype fp32 --device cuda:0 --processor-device cuda:0
```



#### Model Summary
<!-- DType & Device  -->
| Model | Dynamic Batch | TensorRT(FP16)  | FP32 | FP16 | Q8| Q4 | Q4f16| BNB4 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RT-DETR | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| RF-DETR | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| DEIM | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| D-FINE | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |



