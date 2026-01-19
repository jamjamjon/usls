# BiRefNet Examples

This directory contains examples for using BiRefNet (Bimodal Referencing Network) for various image segmentation tasks including portrait matting, camouflaged object detection, and general segmentation.

## Overview

BiRefNet is an advanced image segmentation model that achieves high-precision segmentation through bimodal referencing mechanisms. The model excels in various segmentation tasks including salient object detection, camouflaged object detection, and portrait matting.

## Available Model Variants

### General Purpose Models
- **general**: General-purpose model for versatile segmentation tasks across different domains
- **general_bb_swin_v1_tiny**: General-purpose model with Swin-V1-Tiny backbone for balanced performance and efficiency (default)
- **hr_general**: High-resolution general segmentation model for processing larger images with fine details
- **lite_general_2k**: Lightweight general model optimized for 2K resolution images with efficient inference

### Specialized Models
- **cod**: Camouflaged Object Detection (COD) model for segmenting objects that blend with their surroundings
- **dis**: Dichotomous Image Segmentation (DIS) model for basic foreground/background separation
- **hrsod_dhu**: High-Resolution Salient Object Detection (HRSOD) model trained on DHU dataset
- **massive**: Massive model trained on multiple datasets including DIS5K and TE datasets for robust performance

### Portrait & Matting Models
- **portrait**: Specialized portrait segmentation model for high-quality portrait background removal
- **matting**: Portrait matting model for precise hair and fine detail preservation in portraits
- **hr_matting**: High-resolution portrait matting model for detailed portrait segmentation on larger images

## Quick Start

### Input/Output
- `--source <SOURCE>`: Input source - image path, folder, or video (default: `images/liuyifei.png`)

## Examples

### hr_matting (fp16)
```bash
cargo run -r -F cuda-full --example birefnet -- --device cuda:2 --processor-device cuda:2 --variant hr_matting --dtype fp16
```

### general_bb_swin_v1_tiny (q4f16)
```bash
cargo run -r -F cuda-full --example birefnet -- --device cuda:2 --processor-device cuda:2 --variant general_bb_swin_v1_tiny --dtype q4f16
```

### cod (fp16)
```bash
cargo run -r -F cuda-full --example birefnet -- --device cuda:2 --processor-device cuda:2 --variant cod --dtype fp16
```

### dis (q4f16)
```bash
cargo run -r -F cuda-full --example birefnet -- --device cuda:2 --processor-device cuda:2 --variant dis --dtype q4f16
```

### hrsod_dhu (fp16)
```bash
cargo run -r -F cuda-full --example birefnet -- --device cuda:2 --processor-device cuda:2 --variant hrsod_dhu --dtype fp16
```

### massive (q4f16)
```bash
cargo run -r -F cuda-full --example birefnet -- --device cuda:2 --processor-device cuda:2 --variant massive --dtype q4f16
```

### general (fp16)
```bash
cargo run -r -F cuda-full --example birefnet -- --device cuda:2 --processor-device cuda:2 --variant general --dtype fp16
```

### portrait (q4f16)
```bash
cargo run -r -F cuda-full --example birefnet -- --device cuda:2 --processor-device cuda:2 --variant portrait --dtype q4f16
```

### matting (fp16)
```bash
cargo run -r -F cuda-full --example birefnet -- --device cuda:2 --processor-device cuda:2 --variant matting --dtype fp16
```

### Custom model file (fp16)
```bash
cargo run -r -F cuda-full --example birefnet -- --device cuda:2 --processor-device cuda:2 --model ../biRefNet-qnt/COD-epoch-125-fp16.onnx --dtype fp16
```

### Custom model file (q4f16)
```bash
cargo run -r -F cuda-full --example birefnet -- --device cuda:2 --processor-device cuda:2 --model ../biRefNet-qnt/general-bb-swin-v1-tiny-epoch-232-q4f16.onnx --dtype q4f16
```