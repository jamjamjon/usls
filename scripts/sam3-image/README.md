# SAM3 Image ONNX Export & Inference

## Updates

| Version | ONNX Files | Resolution | Notes |
|---------|------------|------------|-------|
| **v1(4 onnx)** | Vision-encoder<br>Text-encoder<br>Geometry-encoder<br>Decoder | 1008×1008 | Original architecture |
| **v2(3 onnx)** | Vision-encoder<br>Text-encoder<br>Geo-Encoder-Mask-Decoder | 1008×1008 | Geometry integrated into Decoder |


## Reference
- **Exported ONNX models:** https://github.com/jamjamjon/assets/releases/tag/sam3
- **Sam3Image Demo (text & bboxes prompts):** [sam3-image](../../examples/open-set-segmentation/README.md#sam3-image)
- **Sam3Tracker Demo (points & bboxes prompts):** [sam3-tracker](../../examples/image-segmentation/README.md#sam3-tracker)
- **Sam3Image Implementation:** [sam3_image](../../src/models/vlm/sam3_image)
- **Sam3Tracker Implementation:** [sam3_tracker](../../src/models/vision/sam3_tracker)


## Export ONNX Models

### v2 (3 ONNX, 1008×1008) - Recommended
```bash
uv run export_v2.py --all \
  --model-path /path/to/sam3-models \
  --output-dir onnx-models-v2 \
  --device cuda \
  --image-height 1008 --image-width 1008
```


### v1 (4 ONNX, 1008×1008)

<details>
<summary>Click to expand v1 export commands</summary>

```bash
uv run export.py --all --model-path /path/to/sam3-models --output-dir onnx-models
```

</details>



## Inference Code

### Python (v2, 1008×1008) - Recommended

```bash
# Text prompt
uv run inference_v2.py \
  --image ../../assets/kids.jpg \
  --text "shoe" \
  --model-dir ./onnx-models-v2 \
    --tokenizer /path/to/tokenizer.json \
  --output output-text-v2.png \
  --device cuda \
  --image-height 1008 --image-width 1008

# Box prompt (xywh format: x,y,w,h)
uv run inference_v2.py \
  --image ../../assets/kids.jpg \
  --boxes "pos:480,290,110,360" \
  --model-dir ./onnx-models-v2 \
    --tokenizer /path/to/tokenizer.json \
    --tokenizer /path/to/tokenizer.json \
  --output output-box-v2.png \
  --device cuda \
  --image-height 1008 --image-width 1008

# Box prompt with positive + negative
uv run inference_v2.py \
  --image ../../assets/kids.jpg \
  --boxes "pos:480,290,110,360;neg:370,280,115,375" \
  --model-dir ./onnx-models-v2 \
    --tokenizer /path/to/tokenizer.json \
  --output output-box-posneg-v2.png \
  --device cuda \
  --image-height 1008 --image-width 1008


# Text + Negative box (mixed prompt)
uv run inference_v2.py \
    --image ../../assets/oven.jpg \
    --text "handle" \
    --boxes "neg:40,183,278,21" \
    --model-dir ./onnx-models-v2 \
    --tokenizer /path/to/tokenizer.json \
    --output output-text-box-v2.png \
    --device cuda \
    --image-height 1008 --image-width 1008

```

### Python (v1, 1008×1008)

<details>
<summary>Click to expand v1 inference commands</summary>

```bash
# Text prompt
uv run inference.py \
    --image ../../assets/kids.jpg \
    --text "shoe" \
    --model-dir ./onnx-models \
    --tokenizer /path/to/tokenizer.json \
    --output output-text-v1.png

# Box prompt (xywh format: x,y,w,h)
uv run inference.py \
    --image ../../assets/kids.jpg \
    --boxes "pos:480,290,110,360" \
    --model-dir ./onnx-models \
    --tokenizer /path/to/tokenizer.json \
    --output output-box-v1.png

# Positive + Negative box
uv run inference.py \
    --image ../../assets/kids.jpg \
    --boxes "pos:480,290,110,360;neg:370,280,115,375" \
    --model-dir ./onnx-models \
    --tokenizer /path/to/tokenizer.json \
    --output output-box-posneg-v1.png

# Text + Negative box (mixed prompt)
uv run inference.py \
    --image ../../assets/oven.jpg \
    --text "handle" \
    --boxes "neg:40,183,278,21" \
    --model-dir ./onnx-models \
    --tokenizer /path/to/tokenizer.json \
    --output output-text-box-v1.png
```

</details>



## TensorRT Conversion

Choose `optShapes` and `maxShapes` according to your GPU memory.

### v2 (1008×1008) - Recommended

```bash
# Vision Encoder
/home/qweasd/Documents/TensorRT-10.11.0.33/bin/trtexec --fp16 --onnx=onnx-models-v2/vision-encoder.onnx \
    --minShapes=images:1x3x1008x1008 \
    --optShapes=images:4x3x1008x1008 \
    --maxShapes=images:8x3x1008x1008 \
    --saveEngine=onnx-models-v2/vision-encoder.engine

# Text Encoder
/home/qweasd/Documents/TensorRT-10.11.0.33/bin/trtexec --fp16 --onnx=onnx-models-v2/text-encoder.onnx \
    --minShapes=input_ids:1x32,attention_mask:1x32 \
    --optShapes=input_ids:4x32,attention_mask:4x32 \
    --maxShapes=input_ids:8x32,attention_mask:8x32 \
    --saveEngine=onnx-models-v2/text-encoder.engine

# Decoder (with integrated Geometry Encoder)
/home/qweasd/Documents/TensorRT-10.11.0.33/bin/trtexec --fp16 --onnx=onnx-models-v2/decoder.onnx \
    --minShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,text_features:1x32x256,text_mask:1x32,input_boxes:1x1x4,input_boxes_labels:1x1 \
    --optShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,text_features:1x32x256,text_mask:1x32,input_boxes:1x8x4,input_boxes_labels:1x8 \
    --maxShapes=fpn_feat_0:8x256x288x288,fpn_feat_1:8x256x144x144,fpn_feat_2:8x256x72x72,fpn_pos_2:8x256x72x72,text_features:8x32x256,text_mask:8x32,input_boxes:8x20x4,input_boxes_labels:8x20 \
    --saveEngine=onnx-models-v2/decoder.engine
```

### v1 (1008×1008, 4 ONNX)

<details>
<summary>Click to expand v1 TensorRT conversion commands</summary>

```bash
# Vision Encoder
trtexec --fp16 --onnx=onnx-models/vision-encoder.onnx \
    --minShapes=images:1x3x1008x1008 \
    --optShapes=images:4x3x1008x1008 \
    --maxShapes=images:8x3x1008x1008 \
    --saveEngine=onnx-models/vision-encoder.engine

# Text Encoder
trtexec --fp16 --onnx=onnx-models/text-encoder.onnx \
    --minShapes=input_ids:1x32,attention_mask:1x32 \
    --optShapes=input_ids:4x32,attention_mask:4x32 \
    --maxShapes=input_ids:8x32,attention_mask:8x32 \
    --saveEngine=onnx-models/text-encoder.engine

# Geometry Encoder
trtexec --fp16 --onnx=onnx-models/geometry-encoder.onnx \
    --minShapes=input_boxes:1x1x4,input_boxes_labels:1x1,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72 \
    --optShapes=input_boxes:1x8x4,input_boxes_labels:1x8,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72 \
    --maxShapes=input_boxes:8x20x4,input_boxes_labels:8x20,fpn_feat_2:8x256x72x72,fpn_pos_2:8x256x72x72 \
    --saveEngine=onnx-models/geometry-encoder.engine

# Decoder
trtexec --fp16 --onnx=onnx-models/decoder.onnx \
    --minShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x1x256,prompt_mask:1x1 \
    --optShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x33x256,prompt_mask:1x33 \
    --maxShapes=fpn_feat_0:8x256x288x288,fpn_feat_1:8x256x144x144,fpn_feat_2:8x256x72x72,fpn_pos_2:8x256x72x72,prompt_features:8x60x256,prompt_mask:8x60 \
    --saveEngine=onnx-models/decoder.engine
```

</details>

## ONNX Model Specifications

All models support dynamic batch processing.

### v2 (3 ONNX, 1008×1008)

**Vision Encoder**
```
Inputs:
  images                [batch, 3, 1008, 1008]    FLOAT

Outputs:
  fpn_feat_0            [batch, 256, 288, 288]    FLOAT
  fpn_feat_1            [batch, 256, 144, 144]    FLOAT
  fpn_feat_2            [batch, 256, 72, 72]      FLOAT
  fpn_pos_2             [batch, 256, 72, 72]      FLOAT
```

**Text Encoder**
```
Inputs:
  input_ids             [batch, 32]               INT64
  attention_mask        [batch, 32]               INT64

Outputs:
  text_features         [batch, 32, 256]          FLOAT
  text_mask             [batch, 32]               BOOL
```

**Decoder (with integrated Geometry Encoder)**
```
Inputs:
  fpn_feat_0            [batch, 256, 288, 288]    FLOAT
  fpn_feat_1            [batch, 256, 144, 144]    FLOAT
  fpn_feat_2            [batch, 256, 72, 72]      FLOAT
  fpn_pos_2             [batch, 256, 72, 72]      FLOAT
  text_features         [batch, 32, 256]          FLOAT
  text_mask             [batch, 32]               BOOL
  input_boxes           [batch, num_boxes, 4]     FLOAT
  input_boxes_labels    [batch, num_boxes]        INT64  (1=pos, 0=neg, -10=ignore)

Outputs:
  pred_masks            [batch, 200, 288, 288]    FLOAT
  pred_boxes            [batch, 200, 4]           FLOAT
  pred_logits           [batch, 200]              FLOAT
  presence_logits       [batch, 1]                FLOAT
```


### v1 (4 ONNX, 1008×1008)

<details>
<summary>Click to expand v1 TensorRT conversion commands</summary>


**Vision Encoder** - Same as v2 1008×1008

**Text Encoder** - Same as v2

**Geometry Encoder**
```
Inputs:
  input_boxes           [batch, num_boxes, 4]     FLOAT
  input_boxes_labels    [batch, num_boxes]        INT64
  fpn_feat_2            [batch, 256, 72, 72]      FLOAT
  fpn_pos_2             [batch, 256, 72, 72]      FLOAT

Outputs:
  geometry_features     [batch, num_boxes+1, 256] FLOAT
  geometry_mask         [batch, num_boxes+1]      BOOL
```

**Decoder**
```
Inputs:
  fpn_feat_0            [batch, 256, 288, 288]    FLOAT
  fpn_feat_1            [batch, 256, 144, 144]    FLOAT
  fpn_feat_2            [batch, 256, 72, 72]      FLOAT
  fpn_pos_2             [batch, 256, 72, 72]      FLOAT
  prompt_features       [batch, prompt_len, 256]  FLOAT  (text + geometry concatenated)
  prompt_mask           [batch, prompt_len]       BOOL

Outputs:
  pred_masks            [batch, 200, 288, 288]    FLOAT
  pred_boxes            [batch, 200, 4]           FLOAT
  pred_logits           [batch, 200]              FLOAT
  presence_logits       [batch, 1]                FLOAT
```

</details>
