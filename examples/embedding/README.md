# Embedding Examples

This directory contains examples for embedding models that extract feature representations from images and text.

## Models

### CLIP (Contrastive Language-Image Pre-training)
Vision-language model for image and text embeddings.

**Variants:**
- `clip-b16` - CLIP ViT-B/16
- `clip-b32` - CLIP ViT-B/32
- `clip-l14` - CLIP ViT-L/14
- `jina-clip-v1` - Jina CLIP v1
- `jina-clip-v2` - Jina CLIP v2
- `mobileclip-s0` - MobileCLIP S0
- `mobileclip-s1` - MobileCLIP S1
- `mobileclip-s2` - MobileCLIP S2
- `mobileclip-b` - MobileCLIP B
- `mobileclip-blt` - MobileCLIP BLT
- `mobileclip2-s0` - MobileCLIP2 S0 (default)
- `mobileclip2-s2` - MobileCLIP2 S2
- `mobileclip2-s4` - MobileCLIP2 S4
- `mobileclip2-b` - MobileCLIP2 B
- `mobileclip2-l14` - MobileCLIP2 L14

**Usage:**
```bash
# Using module-specific device/dtype for visual and textual encoders
cargo run -F cuda-full -F vlm --example embedding -- clip --visual-dtype fp16 --visual-device cuda:0 --textual-dtype fp16 --textual-device cuda:0 --processor-device cuda:0 --variant mobileclip2-s0
```

### DINO (Self-Distillation with No Labels)
Self-supervised vision transformer for image embeddings.

**Variants:**
- `v2-s` - DINOv2 Small (default)
- `v2-b` - DINOv2 Base
- `v3-s` - DINOv3 ViT-S/16 LVD-1689M
- `v3-s-plus` - DINOv3 ViT-S/16+ LVD-1689M
- `v3-b` - DINOv3 ViT-B/16 LVD-1689M
- `v3-l` - DINOv3 ViT-L/16 LVD-1689M
- `v3-l-sat493m` - DINOv3 ViT-L/16 SAT-493M
- `v3-h-plus` - DINOv3 ViT-H/16+ LVD-1689M

**Usage:**
```bash
cargo run -F cuda-full -F vlm --example embedding -- dino   --device cuda --processor-device cuda --dtype q4f16 --variant v3-s --batch 2
```
