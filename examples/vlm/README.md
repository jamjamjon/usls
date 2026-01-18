# Vision-Language Model (VLM) Examples

This directory contains examples for vision-language models that combine visual and textual understanding.

## Models

### BLIP (Bootstrapping Language-Image Pre-training)
Image captioning model.

**Variants:**
- `v1-base-caption` - BLIP v1 Base Caption (default)

**Usage:**
```bash
# Unconditional caption (using module-specific device/dtype)
cargo run -F cuda-full -F vlm --example vlm -- blip --visual-dtype fp32 --visual-device cuda:0 --textual-dtype fp32 --textual-device cuda:0 --processor-device cuda:0 --source ./assets/bus.jpg

# Conditional caption
cargo run -F cuda-full -F vlm --example vlm -- blip --visual-dtype fp32 --visual-device cuda:0 --textual-dtype fp32 --textual-device cuda:0 --processor-device cuda:0 --source ./assets/bus.jpg --prompt "this image depicts"
```

### FastVLM
Fast vision-language model for image understanding.

**Scales:**
- `0.5b` - 0.5 billion parameters (default)

**Usage:**
```bash
cargo run -F cuda-full -F vlm --example vlm  -- fastvlm  --device cuda:0 --processor-device cuda:0 --dtype q4f16 --source ./assets/bus.jpg --scale 0.5b --prompt "Describe the image in detail."

```

### Florence2
Microsoft's Florence-2 unified vision foundation model.

**Scales:**
- `base` - Base model (default)
- `large` - Large model
- `large-ft` - Large fine-tuned model

**Tasks:**
- `Caption: 0` - Brief caption
- `Caption: 1` - Detailed caption
- `Caption: 2` - More detailed caption
- `Ocr` - Optical character recognition
- `ObjectDetection` - Detect objects
- `OpenSetDetection: <query>` - Detect specific objects
- `RegionProposal` - Propose regions
- And more...

**Usage:**
```bash
# Using module-specific device/dtype for visual, textual encoder, and textual decoder
cargo run -r -F cuda-full -F vlm --example vlm -- florence2 --visual-dtype fp16 --visual-device cuda:0 --textual-encoder-dtype fp16 --textual-encoder-device cuda:0 --textual-decoder-dtype fp16 --textual-decoder-device cuda:0 --processor-device cuda:0 --source ./assets/bus.jpg --scale base --task "Caption: 0"

# TODO:
# let tasks = [
#         // w inputs
#         Task::Caption(0),
#         Task::Caption(1),
#         Task::Caption(2),
#         Task::Ocr,
#         // Task::OcrWithRegion,
#         Task::RegionProposal,
#         Task::ObjectDetection,
#         Task::DenseRegionCaption,
#         // w/o inputs
#         Task::OpenSetDetection("a vehicle".into()),
#         Task::CaptionToPhraseGrounding(
#             "A vehicle with two wheels parked in front of a building.".into(),
#         ),
#         Task::ReferringExpressionSegmentation("a vehicle".into()),
#         Task::RegionToSegmentation(
#             // 31, 156, 581, 373,  // car
#             449, 270, 556, 372, // wheel
#         ),
#         Task::RegionToCategory(
#             // 31, 156, 581, 373,
#             449, 270, 556, 372,
#         ),
#         Task::RegionToDescription(
#             // 31, 156, 581, 373,
#             449, 270, 556, 372,
#         ),
#     ];

```

### Moondream2
Compact vision-language model for various vision tasks.

**Scales:**
- `0.5b` - 0.5 billion parameters (default)
- `2b` - 2 billion parameters

**Tasks:**
- `Caption: 0` - Image captioning
- `Vqa: <question>` - Visual question answering
- `OpenSetDetection: <query>` - Open-set object detection
- `OpenSetKeypointsDetection: <query>` - Open-set keypoint detection

**Usage:**
```bash
# Using module-specific device/dtype for all 8 modules
cargo run -F cuda-full -F vlm --example vlm -- --source ./assets/bus.jpg moondream2 --scale 0.5b --visual-encoder-dtype int8 --visual-encoder-device cuda:0 --visual-projection-dtype int8 --visual-projection-device cuda:0 --textual-encoder-dtype int8 --textual-encoder-device cuda:0 --textual-decoder-dtype int8 --textual-decoder-device cuda:0 --coord-encoder-dtype int8 --coord-encoder-device cuda:0 --coord-decoder-dtype int8 --coord-decoder-device cuda:0 --size-encoder-dtype int8 --size-encoder-device cuda:0 --size-decoder-dtype int8 --size-decoder-device cuda:0 --processor-device cuda:0 --task "Caption: 0"

# VQA example
cargo run -F cuda-full -F vlm --example vlm -- --source ./assets/bus.jpg moondream2 --visual-encoder-dtype int8 --visual-encoder-device cuda:0 --textual-decoder-dtype int8 --textual-decoder-device cuda:0 --processor-device cuda:0 --task "Vqa: What is in the image?"
```

### SmolVLM
Small vision-language model optimized for efficiency.

**Scales:**
- `256m` - 256 million parameters (default)
- `500m` - 500 million parameters

**Versions:**
- `1` - Version 1
- `2` - Version 2 (default)

**Usage:**
```bash
cargo run -F vlm --example vlm -- --source ./assets/bus.jpg smolvlm --scale 256m --ver 2 --prompt "Can you describe this image?"
```
