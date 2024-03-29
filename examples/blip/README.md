This demo shows how to use [BLIP](https://arxiv.org/abs/2201.12086) to do conditional or unconditional image captioning.


## Quick Start

```shell
cargo run -r --example blip
```

## Or you can manully


### 1. Donwload CLIP ONNX Model

[blip-visual-base](https://github.com/jamjamjon/assets/releases/download/v0.0.1/blip-visual-base.onnx)  
[blip-textual-base](https://github.com/jamjamjon/assets/releases/download/v0.0.1/blip-textual-base.onnx)


### 2. Specify the ONNX model path in `main.rs`

```Rust
    // visual
    let options_visual = Options::default()
        .with_model("VISUAL_MODEL")   // <= modify this
        .with_profile(false);

    // textual
    let options_textual = Options::default()
        .with_model("TEXTUAL_MODEL")  // <= modify this
        .with_profile(false);

```

### 3. Then, run

```bash
cargo run -r --example blip
```


## Results

```shell
[Unconditional image captioning]: a group of people walking around a bus
[Conditional image captioning]: three man walking in front of a bus
```

## TODO

* [ ] text decode with Top-p sample
* [ ] VQA
* [ ] Retrival
* [ ] TensorRT support for textual model
