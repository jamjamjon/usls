# SAM3: Segment Anything with Concepts

A powerful multimodal segmentation model supporting text, bounding box, and combined prompts.

## References

- Official: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- ONNX Models: [Download](https://github.com/jamjamjon/assets/releases/tag/sam3)

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
  fpn_feat_3            [batch, 256, 36, 36]      FLOAT
  fpn_pos_0             [batch, 256, 288, 288]    FLOAT
  fpn_pos_1             [batch, 256, 144, 144]    FLOAT
  fpn_pos_2             [batch, 256, 72, 72]      FLOAT
  fpn_pos_3             [batch, 256, 36, 36]      FLOAT
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
  fpn_feat              [batch, 256, 72, 72]      FLOAT
  fpn_pos               [batch, 256, 72, 72]      FLOAT

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

## TensorRT Conversion

### Vision Encoder
```bash
trtexec --onnx=sam3_vision_encoder.onnx \
  --minShapes=images:1x3x1008x1008 \
  --optShapes=images:4x3x1008x1008 \
  --maxShapes=images:8x3x1008x1008 \
  --saveEngine=sam3_vision_encoder.engine
```

### Text Encoder
```bash
trtexec --onnx=sam3_text_encoder.onnx \
  --minShapes=input_ids:1x32,attention_mask:1x32 \
  --optShapes=input_ids:4x32,attention_mask:4x32 \
  --maxShapes=input_ids:8x32,attention_mask:8x32 \
  --saveEngine=sam3_text_encoder.engine
```

### Geometry Encoder
```bash
trtexec --onnx=sam3_geometry_encoder.onnx \
  --minShapes=input_boxes:1x1x4,input_boxes_labels:1x1,fpn_feat:1x256x72x72,fpn_pos:1x256x72x72 \
  --optShapes=input_boxes:1x5x4,input_boxes_labels:1x5,fpn_feat:1x256x72x72,fpn_pos:1x256x72x72 \
  --maxShapes=input_boxes:8x20x4,input_boxes_labels:8x20,fpn_feat:8x256x72x72,fpn_pos:8x256x72x72 \
  --saveEngine=sam3_geometry_encoder.engine
```

### Decoder
```bash
trtexec --onnx=sam3_decoder.onnx \
  --minShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x1x256,prompt_mask:1x1 \
  --optShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x33x256,prompt_mask:1x33 \
  --maxShapes=fpn_feat_0:8x256x288x288,fpn_feat_1:8x256x144x144,fpn_feat_2:8x256x72x72,fpn_pos_2:8x256x72x72,prompt_features:8x60x256,prompt_mask:8x60 \
  --saveEngine=sam3_decoder.engine
```


## Example

See [examples/sam3](../../../examples/sam3)
