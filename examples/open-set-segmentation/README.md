# Open-Set Segmentation Examples

This directory contains examples for open-set segmentation models, which can segment objects based on arbitrary text prompts or visual cues.

## Models

### SAM3-Image
The latest generation of Segment Anything Model (SAM3) optimized for image-based open-set segmentation.
- Supports both text prompts and geometry (point/box) prompts.
- Advanced hierarchical feature extraction.

**Note**: For SAM3 with box/point prompts, see the [sam3-tracker example](../image-segmentation/README.md#sam3-tracker).

### YOLOEPromptBased
YOLOE with prompt support for flexible object detection and segmentation.
- `Visual`: Uses a visual prompt (image + bounding box) to find similar objects.
- `Textual`: Uses text descriptions to segment objects.


## Examples


### yoloe-prompt-based

#### # Visual prompt

**One image, two boxes prompt**
```bash
cargo run -r -F cuda-full -F vlm --example open-set-segmentation -- yoloe-prompt-based --model-dtype fp32 --model-device cuda --visual-encoder-dtype fp32 --visual-encoder-device cuda --processor-device cuda --ver 26 --scale s --prompt-image ./assets/bus.jpg -p "xyxy:221.52, 405.8, 344.98, 857.54, person" -p "xyxy:25.519104, 234.08078, 789.1692, 737.5875, bus" --source ../coco2017val/coco/images/val2017 --batch 1
```

#### Textual prompt

```bash
cargo run -r -F cuda-full -F vlm --example open-set-segmentation -- yoloe-prompt-based \
--model-dtype fp32 --model-device cuda:0 \
--textual-encoder-dtype f16 --textual-encoder-device cuda:0 \
--processor-device cuda:0 \
--ver 26 --scale s \
--source ../coco2017val/coco/images/val2017 --batch 1 \
-p person -p bus 
```


### SAM3-Image

> **Note**: Now sam3image only uses 3 ONNX models (Vision, Text, Decoder with integrated Geometry). No separate geometry encoder needed.

#### Single text prompt (Use Cuda)

```bash
cargo run -F cuda-full -F vlm --example open-set-segmentation -- sam3-image \
--visual-encoder-dtype q4f16 --visual-encoder-device cuda:0 \
--textual-encoder-dtype q4f16 --textual-encoder-device cuda:0 \
--decoder-dtype q4f16 --decoder-device cuda:0 \
--processor-device cuda:0 \
--source ./assets/kids.jpg \
-p shoe
``` 

#### Single text prompt (Use TensorRT)

**Note**: Use `--visual-encoder-dtype f32` for better performance, or `--visual-encoder-dtype f16` if VRAM is insufficient. If you encounter conversion failures or memory issues, use `--trt_max_workspace_size` to increase the TensorRT workspace size (e.g., `8589934592` for 8GB).

```bash
cargo run -r -F tensorrt-full -F vlm --example open-set-segmentation -- sam3-image \
--visual-encoder-dtype f16 --visual-encoder-device tensorrt:0 \
--textual-encoder-dtype f16 --textual-encoder-device tensorrt:0 \
--decoder-dtype f16 --decoder-device tensorrt:0 \
--processor-device cuda:0 \
--source ./assets
-p person
``` 

#### Multiple text prompts

```bash
cargo run -F cuda-full -F vlm --example open-set-segmentation -- sam3-image \
--visual-encoder-dtype q4f16 --visual-encoder-device cuda:0 \
--textual-encoder-dtype q4f16 --textual-encoder-device cuda:0 \
--decoder-dtype q4f16 --decoder-device cuda:0 \
--processor-device cuda:0 \
--source "assets/kids.jpg" \
-p shoe \
-p "person in blue vest"
```

#### Multiple prompts on multiple images
```bash
cargo run -F cuda-full -F vlm --example open-set-segmentation -- sam3-image \
--visual-encoder-dtype q4f16 --visual-encoder-device cuda:0 \
--textual-encoder-dtype q4f16 --textual-encoder-device cuda:0 \
--decoder-dtype q4f16 --decoder-device cuda:0 \
--processor-device cuda:0 \
--source "assets/kids.jpg | assets/bus.jpg" -p bus -p cat -p shoe -p cellphone -p person -p "short hair"
```

#### Visual prompt (bbox)

```bash
cargo run -F cuda-full -F vlm --example open-set-segmentation -- sam3-image \
--visual-encoder-dtype q4f16 --visual-encoder-device cuda:0 \
--textual-encoder-dtype q4f16 --textual-encoder-device cuda:0 \
--decoder-dtype q4f16 --decoder-device cuda:0 \
--processor-device cuda:0 \
--source assets/kids.jpg -p "pos:480,290,110,360"
```

#### Visual prompt: multi-boxes prompting(with positive and negative boxes)

```bash
cargo run -F cuda-full -F vlm --example open-set-segmentation -- sam3-image \
--visual-encoder-dtype q4f16 --visual-encoder-device cuda:0 \
--textual-encoder-dtype q4f16 --textual-encoder-device cuda:0 \
--decoder-dtype q4f16 --decoder-device cuda:0 \
--processor-device cuda:0 \
--source assets/kids.jpg -p "pos:480,290,110,360;neg:370,280,115,375"
```

#### Text + negative box

```bash
cargo run -F cuda-full -F vlm --example open-set-segmentation -- sam3-image \
--visual-encoder-dtype q4f16 --visual-encoder-device cuda:0 \
--textual-encoder-dtype q4f16 --textual-encoder-device cuda:0 \
--decoder-dtype q4f16 --decoder-device cuda:0 \
--processor-device cuda:0 \
--source assets/oven.jpg \
-p "handle;neg:40,183,278,21"
```

#### Prompt Format

| Format | Description | Example |
|--------|-------------|---------|
| `text` | Text description | `-p "cat"` |
| `pos:x,y` | Positive point (2 coords) | `-p "pos:500,375"` |
| `neg:x,y` | Negative point (2 coords) | `-p "neg:300,400"` |
| `pos:x,y,w,h` | Positive box (4 coords) | `-p "pos:480,290,110,360"` |
| `neg:x,y,w,h` | Negative box (4 coords) | `-p "neg:370,280,115,375"` |
| `text;geo;...` | Text + geometry | `-p "shoe;pos:480,290,110,360"` |

#### Parsing Rules

1. First part without `pos:`/`neg:` prefix → **text prompt**
2. Parts with `pos:`/`neg:` prefix → **geometry** (point or box)
3. **2 coords** → point, **4 coords** → box (xywh)
4. If only geometry (no text), **"visual"** is auto-set

**Examples:**
- `-p "cat"` → text="cat", no geometry
- `-p "pos:480,290,110,360"` → text="visual" (auto), 1 positive box
- `-p "shoe;pos:480,290,110,360"` → text="shoe", 1 positive box
- `-p "pos:500,375;neg:300,400"` → text="visual" (auto), 1 pos point + 1 neg point




## Results

|model|demo|
|---|---|
|sam3-image multi-text|![](https://github.com/jamjamjon/assets/releases/download/sam3/demo-kids.jpg)|
|sam3-image multi-text|![](https://github.com/jamjamjon/assets/releases/download/sam3/demo-bus.jpg)|
|sam3-image visual-pos|![](https://github.com/jamjamjon/assets/releases/download/sam3/demo-visual-pos.jpg)|
|sam3-image visual-pos-neg|![](https://github.com/jamjamjon/assets/releases/download/sam3/demo-visual-pos-neg.jpg)|
|yoloe-text-prompt|![](https://github.com/jamjamjon/assets/releases/download/yoloe/demo-text-bus-person.jpg)|
|yoloe-visual-prompt|![](https://github.com/jamjamjon/assets/releases/download/yoloe/demo-visual-prompt.jpg)|
