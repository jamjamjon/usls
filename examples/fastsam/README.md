## Quick Start

```shell
cargo run -r --example fastsam
```

## Donwload or export ONNX Model

- **Export**  

    ```bash
    pip install -U ultralytics
    yolo export model=FastSAM-s.pt format=onnx simplify dynamic
    ```

- **Download**  

    [FastSAM-s-dyn-f16](https://github.com/jamjamjon/assets/releases/download/v0.0.1/FastSAM-s-dyn-f16.onnx)


## Results

![](./demo.png)
