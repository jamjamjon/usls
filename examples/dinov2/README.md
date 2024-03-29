This demo showcases how to use `DINOv2` to compute image similarity, applicable for image-to-image retrieval tasks.

## Quick Start

```shell
cargo run -r --example dinov2
```

## Or you can manully

### 1.Donwload DINOv2 ONNX Model

[dinov2-s14](https://github.com/jamjamjon/assets/releases/download/v0.0.1/dinov2-s14.onnx)
[dinov2-s14-dyn](https://github.com/jamjamjon/assets/releases/download/v0.0.1/dinov2-s14-dyn.onnx)
[dinov2-s14-dyn-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/dinov2-s14-dyn-f16.onnx)

[dinov2-b14](https://github.com/jamjamjon/assets/releases/download/v0.0.1/dinov2-b14.onnx)
[dinov2-b14-dyn](https://github.com/jamjamjon/assets/releases/download/v0.0.1/dinov2-b14-dyn.onnx)
[dinov2-b14-dyn-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/dinov2-b14-dyn-f16.onnx)

### 2. Specify the ONNX model path in `main.rs`

```Rust
let options = Options::default()
    .with_model("ONNX_PATH")    // <= modify this
    .with_profile(false);

// build index
let options = IndexOptions {
    dimensions: 384, // 768 for vitb; 384 for vits
    metric: MetricKind::L2sq,
    quantization: ScalarKind::F16,
    ..Default::default()
};
```

### 3. Then, run

```bash
cargo run -r --example dinov2
```

## Results

```shell
Top-1 distance: 0.0 => "./examples/dinov2/images/bus.jpg"
Top-2 distance: 1.8332717 => "./examples/dinov2/images/dog.png"
Top-3 distance: 1.9672602 => "./examples/dinov2/images/cat.png"
Top-4 distance: 1.978817 => "./examples/dinov2/images/carrot.jpg"
```
