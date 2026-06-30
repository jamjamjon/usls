"""Export the SAM3-LiteText MobileCLIP text encoder to ONNX.

SAM3-LiteText (arXiv:2602.12173) keeps the SAM3 ViT-H vision encoder, geometry
encoder and mask decoder intact and only replaces the text encoder with a
distilled MobileCLIP student. usls therefore reuses the existing SAM3 image
vision/decoder ONNX (jamjamjon/assets `sam3` release) and only needs this
lightweight text encoder.

The exported ONNX is a drop-in replacement for the SAM3 text encoder:
  inputs : input_ids[B, 32], attention_mask[B, 32]
  outputs: text_features[B, 32, 256], text_mask[B, 32]  (bool, True = valid)

Variants (HuggingFace `vil-uob/sam3-litetext-{s0,s1,l}`):
  s0 -> MobileCLIP-S0,  s1 -> MobileCLIP-S1,  l -> MobileCLIP2-L

Usage:
  uv run export_text_encoder.py --model vil-uob/sam3-litetext-s0 --prefix sam3-litetext-s0 --precision fp32
  uv run export_text_encoder.py --model vil-uob/sam3-litetext-s0 --prefix sam3-litetext-s0 --precision fp16

fp16 conversion uses NVIDIA Model Optimizer AutoCast (precision-aware; keeps
numerically unsafe nodes in fp32), which handles MobileCLIP's `.float()`
LayerNorm/Softmax regions that onnxconverter_common cannot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoModel


class LiteTextTextEncoder(nn.Module):
    """Wraps the HF model's text path into a plain-tensor ONNX interface."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        text = self.model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        # pooler_output is the per-token projected features [B, seq, 256];
        # text_mask uses True = valid token (matches the SAM3 decoder).
        return text.pooler_output, attention_mask.bool()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="vil-uob/sam3-litetext-s0")
    ap.add_argument("--out-dir", default="onnx-sam3-litetext")
    ap.add_argument("--prefix", default="sam3-litetext-s0")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--seq", type=int, default=32, help="text context length (fixed by the SAM3 decoder)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-fp16" if args.precision == "fp16" else ""
    out = out_dir / f"{args.prefix}-text-encoder{suffix}.onnx"

    model = AutoModel.from_pretrained(args.model).eval()
    wrapper = LiteTextTextEncoder(model).eval()

    # Dummy prompt "dog": BOS, token, EOS, then EOS-padding to `seq`.
    ids = torch.full((1, args.seq), 49407, dtype=torch.long)
    ids[0, :3] = torch.tensor([49406, 1929, 49407])
    attn = torch.ones(1, args.seq, dtype=torch.long)

    with torch.no_grad():
        ref_tf, ref_tm = wrapper(ids, attn)
    print("torch text_features", tuple(ref_tf.shape), "text_mask", tuple(ref_tm.shape))

    torch.onnx.export(
        wrapper, (ids, attn), str(out),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_features", "text_mask"],
        opset_version=17, do_constant_folding=True, dynamo=False,
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                      "text_features": {0: "batch"}, "text_mask": {0: "batch"}},
    )
    print("exported:", out)

    if args.precision == "fp16":
        from modelopt.onnx.autocast import convert_to_mixed_precision

        model_fp16 = convert_to_mixed_precision(
            str(out), low_precision_type="fp16", keep_io_types=True
        )
        onnx.save(model_fp16, str(out))
        print("converted to fp16 (modelopt AutoCast)")

    # Verify ONNX Runtime matches PyTorch.
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    np_dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
                "tensor(int64)": np.int64, "tensor(bool)": np.bool_}
    feeds = {"input_ids": ids.numpy(), "attention_mask": attn.numpy()}
    cast = {i.name: feeds[i.name].astype(np_dtype[i.type]) for i in sess.get_inputs()}
    got = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, cast)))
    r = ref_tf.numpy().astype(np.float64)
    g = np.asarray(got["text_features"], np.float64)
    diff = np.abs(r - g)
    cos = float(r.ravel() @ g.ravel() / (np.linalg.norm(r.ravel()) * np.linalg.norm(g.ravel()) + 1e-12))
    print(f"verify text_features: max_abs={diff.max():.3e} mean_abs={diff.mean():.3e} cos={cos:.6f}")
    mism = int((np.asarray(got["text_mask"]).astype(bool) != ref_tm.numpy()).sum())
    print(f"verify text_mask: mismatched={mism}")


if __name__ == "__main__":
    main()
