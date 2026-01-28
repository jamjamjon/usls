# VLM Models

**usls** supports state-of-the-art Vision-Language Models (VLM) for tasks like image captioning, visual question answering (VQA), and open-set object detection.

## Multi-Modal Models

| Model | Task / Description | Dynamic Batch | FP32 | FP16 | Q4f16 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| [Florence2](https://arxiv.org/abs/2311.06242) | General Vision Tasks | ✅ | ✅ | ✅ | ❌ |
| [Moondream2](https://github.com/vikhyat/moondream) | VQA, Captioning | ✅ | ❌ | ❌ | ✅ |
| [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | Lightweight VQA | ✅ | ✅ | ❓ | ❓ |
| [BLIP](https://github.com/salesforce/BLIP) | Image Captioning | ✅ | ✅ | ❓ | ❌ |

## Open-Set Detection & Segmentation

| Model | Task / Description | Dynamic Batch | FP16 | Q8 |
| :--- | :--- | :---: | :---: | :---: |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Text-to-Bbox | ✅ | ✅ | ✅ |
| [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) | Open-World Detection | ✅ | ✅ | ❌ |
| [YOLO-World](https://github.com/AILab-CVC/YOLO-World) | Real-time Open-Set | ✅ | ✅ | ✅ |

## Embedding Models

| Model | Task / Description | Dynamic Batch | FP16 |
| :--- | :--- | :---: | :---: |
| [CLIP](https://github.com/openai/CLIP) | Vision-Language Embedding | ✅ | ✅ |
| [jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2) | Enhanced Embedding | ✅ | ✅ |
| [DINOv3](https://github.com/facebookresearch/dinov3) | Vision Features | ✅ | ✅ |

---

*See the [Getting Started](../getting-started/run_demo.md) guide to run these models.*
