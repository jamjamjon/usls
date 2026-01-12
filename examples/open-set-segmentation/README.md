# Open-Set Segmentation Examples

This directory contains examples for open-set segmentation models, which can segment objects based on arbitrary text prompts or visual cues.

## Models

### SAM3-Image
The latest generation of Segment Anything Model (SAM3) optimized for image-based open-set segmentation.
- Supports both text prompts and geometry (point/box) prompts.
- Advanced hierarchical feature extraction.

### YOLOEPrompt
YOLOE with prompt support for flexible object detection and segmentation.
- `Visual`: Uses a visual prompt (image + bounding box) to find similar objects.
- `Textual`: Uses text descriptions to segment objects.


## Examples


### YOLOE-Prompt
```bash
# Visual prompt
cargo run -F cuda-full -F vlm --example open-set-segmentation -- yoloe-prompt --dtype q4f16 --device cuda:0 --processor-device cuda:0 --source ./assets/bus.jpg --kind visual

# Textual prompt
cargo run -F cuda-full -F vlm --example open-set-segmentation -- yoloe-prompt --dtype q4f16 --device cuda:0 --processor-device cuda:0 --kind textual --labels person,bus

```


### SAM3-Image

### Text-based Segmentation
```bash
# Single text prompt
cargo run -F cuda-full -F vlm --example open-set-segmentation -- sam3-image --dtype q4f16 --device cuda:0 --processor-device cuda:0 --source assets/kids.jpg -p shoe 


# Multiple text prompts
cargo run -F cuda-full -F vlm --example open-set-segmentation -- sam3-image --dtype q4f16 --device cuda:0 --processor-device cuda:0 --source assets/kids.jpg -p shoe -p "person in blue vest" 

# Multiple prompts on multiple images
cargo run -F cuda-full -F vlm --example open-set-segmentation -- sam3-image --dtype q4f16 --device cuda:0 --processor-device cuda:0 --source assets/kids.jpg --source assets/bus.jpg -p bus -p cat -p shoe -p cellphone -p person -p "short hair"

# Visual prompt (bbox)
cargo run -F cuda-full -F vlm --example open-set-segmentation -- sam3-image --dtype q4f16 --device cuda:0 --processor-device cuda:0 --source assets/kids.jpg -p "pos:480,290,110,360"

# Visual prompt: multi-boxes prompting(with positive and negative boxes)
cargo run -F cuda-full -F vlm --example open-set-segmentation -- sam3-image --dtype q4f16 --device cuda:0 --processor-device cuda:0 --source assets/kids.jpg -p "pos:480,290,110,360;neg:370,280,115,375"

# Text + negative box
cargo run -F cuda-full -F vlm --example open-set-segmentation -- sam3-image --dtype q4f16 --device cuda:0 --processor-device cuda:0 --source assets/000000136466.jpg -p "handle;neg:40,183,278,21"
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
|yoloe-text-prompt|![](https://github.com/jamjamjon/assets/releases/download/yoloe/demo-text-bus-person.jpg)|
|yoloe-visual-prompt|![](https://github.com/jamjamjon/assets/releases/download/yoloe/demo-visual-prompt.jpg)|
|sam3-image|![](https://github.com/jamjamjon/assets/releases/download/sam3/demo-shoes-person-in-blue-vest.jpg)|
|sam3-image|![](https://github.com/jamjamjon/assets/releases/download/sam3/demo.jpg)|
