# Image Classification Examples

This directory contains examples for image classification using various SOTA models.


## Usage

```bash
# BeiT example
cargo run -F cuda-full -F vlm --example image-classification -- beit    --device cuda --processor-device cuda 

# DeiT example
cargo run -F cuda-full -F vlm --example image-classification -- deit    --device cuda --processor-device cuda 

# MobileOne example
cargo run -F cuda-full -F vlm --example image-classification -- mobileone    --device cuda --processor-device cuda --kind s0

# ConvNeXt example
cargo run -F cuda-full -F vlm --example image-classification -- convnext    --device cuda --processor-device cuda

# FastViT
cargo run -F cuda-full -F vlm --example image-classification -- fastvit    --device cuda --processor-device cuda

# RAM example
cargo run -F cuda-full -F vlm --example image-classification -- ram    --device cuda --processor-device cuda

# RAM++ example
cargo run -F cuda-full -F vlm --example image-classification -- ram    --device cuda --processor-device cuda  --variant ram++

```
