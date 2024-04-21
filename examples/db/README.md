## Quick Start

```shell
cargo run -r --example db
```

## ONNX Model

- [ppocr-v3-db-dyn](https://github.com/jamjamjon/assets/releases/download/v0.0.1/ppocr-v3-db-dyn.onnx)  
- [ppocr-v4-db-dyn](https://github.com/jamjamjon/assets/releases/download/v0.0.1/ppocr-v4-db-dyn.onnx)

### Speed test

| Model           | Image size | TensorRT<br />f16<br />batch=1<br />(ms) | TensorRT<br />f32<br />batch=1<br />(ms) | CUDA<br />f32<br />batch=1<br />(ms) |
| --------------- | ---------- | ---------------------------------------- | ---------------------------------------- | ------------------------------------ |
| ppocr-v3-db-dyn | 640x640    | 1.8585                                   | 2.5739                                   | 4.3314                               |
| ppocr-v4-db-dyn | 640x640    | 2.0507                                   | 2.8264                                   | 6.6064                               |

***Test on RTX3060***

## Results

![](./demo.png)
