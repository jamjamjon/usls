This demo showcases how to use `DINOv2` to compute image similarity, applicable for image-to-image retrieval tasks.

## Quick Start

```shell
cargo run -r --example dinov2
```

## Donwload DINOv2 ONNX Model

- [dinov2-s14](https://github.com/jamjamjon/assets/releases/download/v0.0.1/dinov2-s14.onnx)
- [dinov2-s14-dyn](https://github.com/jamjamjon/assets/releases/download/v0.0.1/dinov2-s14-dyn.onnx)
- [dinov2-s14-dyn-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/dinov2-s14-dyn-f16.onnx)

- [dinov2-b14](https://github.com/jamjamjon/assets/releases/download/v0.0.1/dinov2-b14.onnx)
- [dinov2-b14-dyn](https://github.com/jamjamjon/assets/releases/download/v0.0.1/dinov2-b14-dyn.onnx)



## Results

```shell
Top-1  0.0000000 /home/qweasd/Desktop/usls/examples/dinov2/images/bus.jpg
Top-2  1.9059424 /home/qweasd/Desktop/usls/examples/dinov2/images/1.jpg
Top-3  1.9736203 /home/qweasd/Desktop/usls/examples/dinov2/images/2.jpg
```
