# SAM3 Image ONNX Export & Inference

## Export ONNX Models

```bash
uv run export.py --all --model-path /path/to/sam3-models
```

## Inference Code

### Python

```bash
# Text prompt
uv run inference.py \
    --image ../../assets/sam3-demo.jpg \
    --text "shoe" \
    --model-dir ./onnx-models \
    --tokenizer /path/to/tokenizer.json \
    --output output-text.png

# Box prompt (xywh format: x,y,w,h)
uv run inference.py \
    --image ../../assets/sam3-demo.jpg \
    --boxes "pos:480,290,110,360" \
    --model-dir ./onnx-models \
    --tokenizer /path/to/tokenizer.json \
    --output output-box.png

# Positive + Negative box
uv run inference.py \
    --image ../../assets/sam3-demo.jpg \
    --boxes "pos:480,290,110,360;neg:370,280,115,375" \
    --model-dir ./onnx-models \
    --tokenizer /path/to/tokenizer.json \
    --output output-box-posneg.png

# Text + Negative box (mixed prompt)
uv run inference.py \
    --image ../../assets/000000136466.jpg \
    --text "handle" \
    --boxes "neg:40,183,278,21" \
    --model-dir ./onnx-models \
    --tokenizer /path/to/tokenizer.json \
    --output output-text-box.png
```

### Rust

See [Rust Implementation](../../src/models/sam3)

## TensorRT Conversion

You should choose `optShapes` and `maxShapes` according to the available GPU memory on your machine

### Vision Encoder
```bash
trtexec --fp16 --onnx=vision-encoder.onnx \
    --minShapes=images:1x3x1008x1008 \
    --optShapes=images:4x3x1008x1008 \
    --maxShapes=images:8x3x1008x1008 \
    --saveEngine=vision-encoder.engine
```

### Text Encoder
```bash
trtexec --fp16 --onnx=text-encoder.onnx \
    --minShapes=input_ids:1x32,attention_mask:1x32 \
    --optShapes=input_ids:4x32,attention_mask:4x32 \
    --maxShapes=input_ids:8x32,attention_mask:8x32 \
    --saveEngine=text-encoder.engine
```

### Geometry Encoder
```bash
trtexec --fp16 --onnx=geometry-encoder.onnx \
    --minShapes=input_boxes:1x1x4,input_boxes_labels:1x1,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72 \
    --optShapes=input_boxes:1x8x4,input_boxes_labels:1x8,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72 \
    --maxShapes=input_boxes:8x20x4,input_boxes_labels:8x20,fpn_feat_2:8x256x72x72,fpn_pos_2:8x256x72x72 \
    --saveEngine=geometry-encoder.engine
```

### Decoder
```bash
trtexec --fp16 --onnx=decoder.onnx \
    --minShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x1x256,prompt_mask:1x1 \
    --optShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x33x256,prompt_mask:1x33 \
    --maxShapes=fpn_feat_0:8x256x288x288,fpn_feat_1:8x256x144x144,fpn_feat_2:8x256x72x72,fpn_pos_2:8x256x72x72,prompt_features:8x60x256,prompt_mask:8x60 \
    --saveEngine=decoder.engine
```

## ONNX Model Specifications

All models support dynamic batch processing.

### Vision Encoder
```
Inputs:
  images                [batch, 3, 1008, 1008]    FLOAT

Outputs:
  fpn_feat_0            [batch, 256, 288, 288]    FLOAT
  fpn_feat_1            [batch, 256, 144, 144]    FLOAT
  fpn_feat_2            [batch, 256, 72, 72]      FLOAT
  fpn_pos_2             [batch, 256, 72, 72]      FLOAT
```

### Text Encoder
```
Inputs:
  input_ids             [batch, 32]               INT64
  attention_mask        [batch, 32]               INT64

Outputs:
  text_features         [batch, 32, 256]          FLOAT
  text_mask             [batch, 32]               BOOL
```

### Geometry Encoder
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

### Decoder
```
Inputs:
  fpn_feat_0            [batch, 256, 288, 288]    FLOAT
  fpn_feat_1            [batch, 256, 144, 144]    FLOAT
  fpn_feat_2            [batch, 256, 72, 72]      FLOAT
  fpn_pos_2             [batch, 256, 72, 72]      FLOAT
  prompt_features       [batch, prompt_len, 256]  FLOAT
  prompt_mask           [batch, prompt_len]       BOOL

Outputs:
  pred_masks            [batch, 200, 288, 288]    FLOAT
  pred_boxes            [batch, 200, 4]           FLOAT
  pred_logits           [batch, 200]              FLOAT
  presence_logits       [batch, 1]                FLOAT
```
