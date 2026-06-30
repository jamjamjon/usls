# SAM3-LiteText text-encoder ONNX export

[SAM3-LiteText](https://huggingface.co/docs/transformers/model_doc/sam3_lite_text)
(arXiv:2602.12173) is SAM3-Image with the heavy text encoder replaced by a
distilled MobileCLIP student; the vision encoder, geometry encoder and mask
decoder are kept intact. The Rust presets reuse the SAM3 image vision/decoder
ONNX and only need this lightweight text encoder.

The exported text encoder is a drop-in replacement for the SAM3 text encoder
(inputs `input_ids[B,32]`, `attention_mask[B,32]`; outputs `text_features[B,32,256]`,
`text_mask[B,32]`).

| variant | HF model | text encoder |
|---|---|---|
| `s0` | `vil-uob/sam3-litetext-s0` | MobileCLIP-S0 |
| `s1` | `vil-uob/sam3-litetext-s1` | MobileCLIP-S1 |
| `l`  | `vil-uob/sam3-litetext-l`  | MobileCLIP2-L |

## Export

```bash
cd scripts/sam3-litetext
uv run export_text_encoder.py --model vil-uob/sam3-litetext-s0 --prefix sam3-litetext-s0 --precision fp32
uv run export_text_encoder.py --model vil-uob/sam3-litetext-s0 --prefix sam3-litetext-s0 --precision fp16
```

Each run writes to `onnx-sam3-litetext/` (override with `--out-dir`) and verifies
ONNX Runtime matches PyTorch. fp16 uses NVIDIA Model Optimizer AutoCast
(`nvidia-modelopt[onnx]`) for precision-aware conversion.
