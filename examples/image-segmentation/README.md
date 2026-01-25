# Image Segmentation Examples

This directory contains examples for various image segmentation models, ranging from general-purpose interactive segmentation (SAM) to specialized tasks like panoptic driving perception (YOLOP).


## Examples

### rfdetr
```bash
cargo run -F cuda-full --example image-segmentation -- rfdetr  --device cuda --processor-device cuda
```

### yoloe-prompt-free
```bash
cargo run  -F cuda-full --example image-segmentation -- yoloe-prompt-free  --device cuda --processor-device cuda --ver 26 --scale m
```

### FastSAM
```bash
cargo run  -F cuda-full --example image-segmentation -- fastsam  --device cuda --processor-device cuda
```

### YOLOP: Driving Perception
```bash
cargo run  -F cuda-full --example image-segmentation -- yolop  --device cuda --processor-device cuda --source images/car-view.jpg
```
### sam

```bash
cargo run  -F cuda-full --example image-segmentation -- sam  --device cuda --processor-device cuda --source images/truck.jpg
```

### sam2

```bash
# Using module-specific device/dtype for encoder and decoder
cargo run -F cuda-full --example image-segmentation -- sam2 --scale t --encoder-dtype fp32 --encoder-device cuda:0 --decoder-dtype fp32 --decoder-device cuda:0 --processor-device cuda:0 --source images/truck.jpg
```

### sam3-tracker

```bash
# Single point (using module-specific device/dtype)
cargo run -F cuda-full --example image-segmentation -- sam3-tracker --vision-dtype q4f16 --vision-device cuda:0 --decoder-dtype fp16 --decoder-device cuda:0 --processor-device cuda:0 --source images/truck.jpg -p "pos:500,375" 

# Two positive points
cargo run -F cuda-full --example image-segmentation -- sam3-tracker --vision-dtype q4f16 --vision-device cuda:0 --decoder-dtype fp16 --decoder-device cuda:0 --processor-device cuda:0 --source images/truck.jpg -p "pos:500,375;pos:1125,625"

# Box prompt (xywh: x, y, width, height)
cargo run -F cuda-full --example image-segmentation -- sam3-tracker --vision-dtype q4f16 --vision-device cuda:0 --decoder-dtype fp16 --decoder-device cuda:0 --processor-device cuda:0 --source images/truck.jpg -p "pos:425,600,275,275"

# Box + negative point
cargo run -F cuda-full --example image-segmentation -- sam3-tracker --vision-dtype q4f16 --vision-device cuda:0 --decoder-dtype fp16 --decoder-device cuda:0 --processor-device cuda:0 --source images/truck.jpg -p "pos:425,600,275,275;neg:575,750"

# Multiple boxes
cargo run -F cuda-full --example image-segmentation -- sam3-tracker --vision-dtype q4f16 --vision-device cuda:0 --decoder-dtype fp16 --decoder-device cuda:0 --processor-device cuda:0 --source images/truck.jpg -p "pos:75,275,1650,575;pos:425,600,275,275;pos:1375,550,275,250;pos:1240,675,160,75"
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