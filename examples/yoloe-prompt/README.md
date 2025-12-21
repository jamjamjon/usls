## Quick Start

```shell
# Textual prompt (CPU)
cargo run -r -F vlm --example yoloe-prompt -- --source ./assets/bus.jpg --visual false

# Visual prompt (TensorRT)
cargo run -r -F yoloe --example yoloe-prompt -F tensorrt -- --source ./assets/bus.jpg --visual true --device tensorrt
```

