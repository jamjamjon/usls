This demo shows how to use [BLIP](https://arxiv.org/abs/2201.12086) to do conditional or unconditional image captioning.

## Quick Start

```shell
cargo run -r --example blip
```

## BLIP ONNX Model

- [blip-visual-base](https://github.com/jamjamjon/assets/releases/download/v0.0.1/blip-visual-base.onnx)  
- [blip-textual-base](https://github.com/jamjamjon/assets/releases/download/v0.0.1/blip-textual-base.onnx)


## Results

```shell
[Unconditional image captioning]: a group of people walking around a bus
[Conditional image captioning]: three man walking in front of a bus
```

## TODO

* [ ] VQA
* [ ] Retrival
* [ ] TensorRT support for textual model
