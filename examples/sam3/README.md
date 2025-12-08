# SAM3 Example

## Quick Start

### Sam3-Image (Text & Box Segmentation)

```bash
# Basic text prompt
cargo run -r -F sam3,cuda --example sam3 -- \
    --device cuda --dtype q4f16 --source assets/sam3-demo.jpg \
    -p shoe 

# Multiple prompts
cargo run -r -F sam3,cuda --example sam3 -- \
    --device cuda --dtype fp16 --source assets/sam3-demo.jpg \
    -p "person in red vest" --show-mask

# Multiple prompts on multiple images
cargo run -r -F sam3,cuda --example sam3 -- \
    --device cuda --dtype q4f16 --vision-batch 2 \
    --source assets/sam3-demo.jpg --source assets/bus.jpg \
    -p bus -p cat -p shoe -p cellphone -p person -p "short hair"

# Visual prompt (bbox)
cargo run -r -F sam3,cuda --example sam3 -- \
    --device cuda --dtype fp16 --source assets/sam3-demo.jpg \
    -p "pos:480,290,110,360"

# Visual prompt: multi-boxes prompting(with positive and negative boxes)
cargo run -r -F sam3 -F cuda --example sam3 -- \
    --device cuda --source ./assets/sam3-demo.jpg \
    -p "pos:480,290,110,360;neg:370,280,115,375"

# Text + negative box
cargo run -r -F sam3,cuda --example sam3 -- \
    --device cuda --source assets/000000136466.jpg \
    -p "handle;neg:40,183,278,21"

```

### Sam3-Tracker (Point & Box Segmentation)

```bash
# Single point
cargo run -r -F sam3,cuda --example sam3 -- --device cuda --task sam3-tracker \
    --source ./assets/truck.jpg -p "pos:500,375"

# Two positive points
cargo run -r -F sam3,cuda --example sam3 -- --device cuda --task sam3-tracker \
    --source ./assets/truck.jpg -p "pos:500,375;pos:1125,625"

# Positive points + negative point
cargo run -r -F sam3,cuda --example sam3 -- --device cuda --task sam3-tracker \
    --source ./assets/truck.jpg -p "pos:1125.,625.;neg:1120,375"

# Box prompt (xywh: x, y, width, height)
cargo run -r -F sam3,cuda --example sam3 -- --device cuda --task sam3-tracker \
    --source ./assets/truck.jpg -p "pos:425,600,275,275"

# Box + negative point
cargo run -r -F sam3,cuda --example sam3 -- --device cuda --task sam3-tracker \
    --source ./assets/truck.jpg -p "pos:425,600,275,275;neg:575,750"  # left wheel + tire

# Multiple boxes
cargo run -r -F sam3,cuda --example sam3 -- --device cuda --task sam3-tracker \
    --source ./assets/truck.jpg \
    -p "pos:75,275,1650,575;pos:425,600,275,275;pos:1375,550,275,250;pos:1240,675,160,75"
    # -p "pos:75,275,1650,575" -p "pos:425,600,275,275" -p "pos:1375,550,275,250" -p "pos:1240,675,160,75"

```

## Prompt Format

| Format | Description | Example |
|--------|-------------|---------|
| `text` | Text description | `-p "cat"` |
| `pos:x,y` | Positive point (2 coords) | `-p "pos:500,375"` |
| `neg:x,y` | Negative point (2 coords) | `-p "neg:300,400"` |
| `pos:x,y,w,h` | Positive box (4 coords) | `-p "pos:480,290,110,360"` |
| `neg:x,y,w,h` | Negative box (4 coords) | `-p "neg:370,280,115,375"` |
| `text;geo;...` | Text + geometry | `-p "shoe;pos:480,290,110,360"` |

### Parsing Rules

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

|  Sam3-Image | Sam3-Tracker | 
|----|----|
|![](https://github.com/jamjamjon/assets/releases/download/sam3/demo2.jpg) |  |
|![](https://github.com/jamjamjon/assets/releases/download/sam3/demo.jpg) |  | 
