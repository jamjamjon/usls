This demo shows how to use [BLIP](https://arxiv.org/abs/2201.12086) to do conditional or unconditional image captioning.

## Quick Start

```shell
cargo run -r --example blip
```

## Results

```shell
[Unconditional]: a group of people walking around a bus
[Conditional]: three man walking in front of a bus
Some(["three man walking in front of a bus"])
```

## TODO

* [ ] Multi-batch inference for image caption
* [ ] VQA
* [ ] Retrival
* [ ] TensorRT support for textual model
