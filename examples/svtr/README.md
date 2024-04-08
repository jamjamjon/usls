## Quick Start

```shell
cargo run -r --example svtr
```

## Or you can manully

### 1. Donwload ONNX Model

[ppocr-v4-server-svtr-ch-dyn](https://github.com/jamjamjon/assets/releases/download/v0.0.1/ppocr-v4-server-svtr-ch-dyn.onnx)  
[ppocr-v4-svtr-ch-dyn](https://github.com/jamjamjon/assets/releases/download/v0.0.1/ppocr-v4-svtr-ch-dyn.onnx)  
[ppocr-v3-svtr-ch-dyn](https://github.com/jamjamjon/assets/releases/download/v0.0.1/ppocr-v3-svtr-ch-dyn.onnx)  

### 2. Specify the ONNX model path in `main.rs`

```Rust
let options = Options::default()
    .with_model("ONNX_PATH")    // <= modify this
```

### 3. Run

```bash
cargo run -r --example svtr
```

### Speed test

| Model                       | Width | TensorRT<br />f16<br />batch=1<br />(ms) | TensorRT<br />f32<br />batch=1<br />(ms) | CUDA<br />f32<br />batch=1<br />(ms) |
| --------------------------- | :---: | :--------------------------------------: | :--------------------------------------: | :----------------------------------: |
| ppocr-v4-server-svtr-ch-dyn | 1500 |                  4.2116                  |                 13.0013                 |               20.8673               |
| ppocr-v4-svtr-ch-dyn        | 1500 |                  2.0435                  |                  3.1959                  |               10.1750               |
| ppocr-v3-svtr-ch-dyn        | 1500 |                  1.8596                  |                  2.9401                  |                6.8210                |

***Test on RTX3060***

## Results

```shell
[Texts] from the background, but also separate text instances which
[Texts] are closely jointed. Some examples are ilustrated in Fig.7.
[Texts] 你有这么高速运转的机械进入中国，记住我给出的原理
```
