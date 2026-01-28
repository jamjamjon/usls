# Vision Models

**usls** supports a wide range of pure vision models for tasks such as object detection, segmentation, pose estimation, and more.

## Object Detection (RT-DETR & Others)

| Model | Task / Description | Dynamic Batch | TensorRT | FP32 | FP16 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR) | Object Detection | ✅ | ✅ | ✅ | ✅ |
| [RF-DETR](https://github.com/roboflow/rf-detr) | Object Detection | ✅ | ✅ | ✅ | ✅ |
| [D-FINE](https://github.com/manhbd-22022602/D-FINE) | Object Detection | ✅ | ❓ | ✅ | ❌ |

## Image Segmentation (SAM & BiRefNet)

| Model | Task / Description | Dynamic Batch | TensorRT | FP32 | FP16 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| [SAM](https://github.com/facebookresearch/segment-anything) | Segment Anything | ✅ | ❓ | ✅ | ❌ |
| [SAM2](https://github.com/facebookresearch/segment-anything-2) | Segment Anything | ✅ | ❓ | ✅ | ❌ |
| [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) | General Segmentation | ✅ | ❓ | ✅ | ✅ |

## Pose Estimation

| Model | Task / Description | Dynamic Batch | TensorRT | FP32 | FP16 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| [RTMPose](https://github.com/open-mmlab/mmpose) | Keypoint Detection | ✅ | ❓ | ✅ | ✅ |
| [DWPose](https://github.com/IDEA-Research/DWPose) | Keypoint Detection | ✅ | ❓ | ✅ | ✅ |

## Image Classification & Tagging

| Model | Task / Description | Dynamic Batch | TensorRT | FP32 | FP16 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| [BEiT](https://github.com/microsoft/unilm/tree/master/beit) | Classification | ✅ | ✅ | ✅ | ✅ |
| [RAM++](https://github.com/xinyu1205/recognize-anything) | Image Tagging | ✅ | ❓ | ✅ | ✅ |

---

*For VLM models, see the [VLM Models](vlm.md) page.*
