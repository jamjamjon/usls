# Embedding Examples

This directory contains examples for embedding models that extract feature representations from images and text.

## Models

### CLIP (Contrastive Language-Image Pre-training)
Vision-language model for image and text embeddings.

**Variants:**

**OpenAI CLIP:**
- `clip-b16` - ViT-B/16 (85M params)
- `clip-b32` - ViT-B/32 (87M params)
- `clip-l14` - ViT-L/14 (304M params)

**Jina CLIP:**
- `jina-clip-v1` - Improved performance, 224x224
- `jina-clip-v2` - 512x512 resolution, better accuracy

**MobileCLIP (Apple):**
- `mobileclip-s0` - Small variant S0
- `mobileclip-s1` - Small variant S1
- `mobileclip-s2` - Small variant S2
- `mobileclip-b` - Base variant
- `mobileclip-blt` - Base with large text encoder

**MobileCLIP v2:**
- `mobileclip2-s0` - Enhanced small S0 (default)
- `mobileclip2-s2` - Enhanced small S2
- `mobileclip2-s4` - Enhanced small S4
- `mobileclip2-b` - Enhanced base
- `mobileclip2-l14` - Enhanced large

**SigLIP (Google DeepMind):**
- `siglip-b16-224` - Base, patch16, 224x224
- `siglip-b16-256` - Base, patch16, 256x256
- `siglip-b16-384` - Base, patch16, 384x384
- `siglip-b16-512` - Base, patch16, 512x512
- `siglip-l16-256` - Large, patch16, 256x256
- `siglip-l16-384` - Large, patch16, 384x384

**SigLIP v2 (Google DeepMind):**
- `siglip2-b16-224` - Base v2, patch16, 224x224
- `siglip2-b16-256` - Base v2, patch16, 256x256
- `siglip2-b16-384` - Base v2, patch16, 384x384
- `siglip2-b16-512` - Base v2, patch16, 512x512
- `siglip2-l16-256` - Large v2, patch16, 256x256
- `siglip2-l16-384` - Large v2, patch16, 384x384
- `siglip2-l16-512` - Large v2, patch16, 512x512
- `siglip2-so400m-patch14-224` - 400M, patch14, 224x224
- `siglip2-so400m-patch14-384` - 400M, patch14, 384x384
- `siglip2-so400m-patch16-256` - 400M, patch16, 256x256
- `siglip2-so400m-patch16-384` - 400M, patch16, 384x384
- `siglip2-so400m-patch16-512` - 400M, patch16, 512x512

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
