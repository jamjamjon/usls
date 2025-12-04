"""
SAM3 ONNX Export Script (Dynamic Batch + TensorRT Compatible)
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from transformers.masking_utils import create_bidirectional_mask
from transformers.models.sam3.modeling_sam3 import Sam3Model


# TensorRT-compatible Position Encoding
def compute_sine_position_encoding(
    shape: tuple,
    device: torch.device,
    dtype: torch.dtype,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    scale: float = 2 * math.pi,
) -> torch.Tensor:
    """Compute sine position encoding (using arange instead of cumsum for TensorRT compatibility)"""
    batch_size, channels, height, width = shape

    y_embed = (
        torch.arange(1, height + 1, dtype=dtype, device=device)
        .view(1, height, 1)
        .expand(batch_size, height, width)
    )
    x_embed = (
        torch.arange(1, width + 1, dtype=dtype, device=device)
        .view(1, 1, width)
        .expand(batch_size, height, width)
    )

    eps = 1e-6
    y_embed = y_embed / (height + eps) * scale
    x_embed = x_embed / (width + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=dtype, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t

    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)

    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class VisionEncoderWrapper(nn.Module):
    """Vision Encoder: ViT backbone + FPN neck (with pre-computed Position Embeddings)"""

    def __init__(self, sam3_model: Sam3Model, device="cpu", image_size=1008):
        super().__init__()

        backbone = sam3_model.vision_encoder.backbone
        self.patch_embeddings = backbone.embeddings.patch_embeddings
        self.dropout = backbone.embeddings.dropout
        self.layer_norm = backbone.layer_norm
        self.layers = backbone.layers
        self.neck = sam3_model.vision_encoder.neck

        patch_size = backbone.config.patch_size
        self.height_patches = image_size // patch_size
        self.width_patches = image_size // patch_size
        hidden_size = backbone.config.hidden_size

        # Pre-compute ViT position embeddings
        orig_pos_embed = backbone.embeddings.position_embeddings.data
        pretrain_size = int(orig_pos_embed.shape[1] ** 0.5)

        pos_embed = orig_pos_embed.reshape(
            1, pretrain_size, pretrain_size, hidden_size
        ).permute(0, 3, 1, 2)
        repeat_h = self.height_patches // pretrain_size + 1
        repeat_w = self.width_patches // pretrain_size + 1
        pos_embed = pos_embed.tile([1, 1, repeat_h, repeat_w])[
            :, :, : self.height_patches, : self.width_patches
        ]
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(
            1, self.height_patches * self.width_patches, hidden_size
        )
        self.register_buffer("vit_pos_embed", pos_embed.to(device))

        # Pre-compute FPN sine position encoding (only need level 2)
        num_pos_feats = sam3_model.vision_encoder.neck.config.fpn_hidden_size // 2
        pos_enc_2 = compute_sine_position_encoding(
            shape=(1, 256, 72, 72),
            device=device,
            dtype=torch.float32,
            num_pos_feats=num_pos_feats,
        )
        self.register_buffer("pos_enc_2", pos_enc_2)

    def forward(self, images: torch.Tensor):
        batch_size = images.shape[0]

        embeddings = self.patch_embeddings(images)
        embeddings = embeddings + self.vit_pos_embed
        embeddings = self.dropout(embeddings)

        hidden_states = embeddings.view(
            batch_size, self.height_patches, self.width_patches, -1
        )
        hidden_states = self.layer_norm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = hidden_states.view(
            batch_size, self.height_patches * self.width_patches, -1
        )
        hidden_states_spatial = hidden_states.view(
            batch_size, self.height_patches, self.width_patches, -1
        ).permute(0, 3, 1, 2)

        fpn_hidden_states, _ = self.neck(hidden_states_spatial)

        # Only return required outputs: fpn_feat_0/1/2 + fpn_pos_2
        return (
            fpn_hidden_states[0],  # [B, 256, 288, 288] for Decoder
            fpn_hidden_states[1],  # [B, 256, 144, 144] for Decoder
            fpn_hidden_states[2],  # [B, 256, 72, 72] for GE & Decoder
            self.pos_enc_2.expand(
                batch_size, -1, -1, -1
            ),  # [B, 256, 72, 72] for GE & Decoder
        )


class TextEncoderWrapper(nn.Module):
    """Text Encoder: CLIP text encoder + projection"""

    def __init__(self, sam3_model: Sam3Model):
        super().__init__()
        self.text_encoder = sam3_model.text_encoder
        self.text_projection = sam3_model.text_projection

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        text_features = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        text_features = self.text_projection(text_features)
        text_mask = attention_mask > 0
        return text_features, text_mask


class GeometryEncoderWrapper(nn.Module):
    """Geometry Encoder (TensorRT compatible: using concat instead of scatter)"""

    def __init__(self, sam3_model: Sam3Model):
        super().__init__()
        self.geometry_encoder = sam3_model.geometry_encoder
        self.hidden_size = self.geometry_encoder.hidden_size
        self.roi_size = self.geometry_encoder.roi_size

    def forward(
        self,
        input_boxes: torch.Tensor,
        input_boxes_labels: torch.Tensor,
        fpn_feat: torch.Tensor,
        fpn_pos: torch.Tensor,
    ):
        batch_size, num_boxes = input_boxes.shape[:2]
        device = input_boxes.device

        box_mask = input_boxes_labels != -10
        box_labels = torch.where(
            input_boxes_labels == -10,
            torch.zeros_like(input_boxes_labels),
            input_boxes_labels,
        )

        ge = self.geometry_encoder

        vision_feats_flat = fpn_feat.flatten(2).transpose(1, 2)
        vision_pos_embeds_flat = fpn_pos.flatten(2).transpose(1, 2)

        img_feats_last = fpn_feat.permute(0, 2, 3, 1)
        normalized_img_feats = ge.vision_layer_norm(img_feats_last).permute(0, 3, 1, 2)

        height, width = normalized_img_feats.shape[-2:]
        boxes_embed = ge.boxes_direct_project(input_boxes)

        # ROI Align (dynamic batch)
        boxes_xyxy = self._box_cxcywh_to_xyxy(input_boxes)
        scale = torch.tensor(
            [width, height, width, height], dtype=boxes_xyxy.dtype, device=device
        )
        boxes_xyxy = boxes_xyxy * scale.view(1, 1, 4)

        batch_indices = (
            torch.arange(batch_size, device=device)
            .view(-1, 1)
            .expand(-1, num_boxes)
            .reshape(-1, 1)
            .float()
        )
        boxes_flat = boxes_xyxy.view(-1, 4)
        boxes_with_batch = torch.cat([batch_indices, boxes_flat], dim=1)

        dtype = (
            torch.float16
            if normalized_img_feats.dtype == torch.bfloat16
            else normalized_img_feats.dtype
        )
        sampled_features = torchvision.ops.roi_align(
            normalized_img_feats.to(dtype), boxes_with_batch.to(dtype), self.roi_size
        ).to(normalized_img_feats.dtype)

        pooled_projection = ge.boxes_pool_project(sampled_features).view(
            batch_size, num_boxes, self.hidden_size
        )
        boxes_embed = boxes_embed + pooled_projection

        # Position encoding
        center_x, center_y = input_boxes[:, :, 0], input_boxes[:, :, 1]
        box_width, box_height = input_boxes[:, :, 2], input_boxes[:, :, 3]
        pos_enc = ge._encode_box_coordinates(
            center_x.flatten(),
            center_y.flatten(),
            box_width.flatten(),
            box_height.flatten(),
        )
        pos_enc = pos_enc.view(batch_size, num_boxes, pos_enc.shape[-1])
        boxes_embed = boxes_embed + ge.boxes_pos_enc_project(pos_enc)

        # Label embeddings + CLS token
        label_embed = ge.label_embed(box_labels.long())
        prompt_embeds = label_embed + boxes_embed
        prompt_mask = box_mask

        cls_embed = ge.cls_embed.weight.view(1, 1, self.hidden_size).expand(
            batch_size, -1, -1
        )
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        prompt_embeds = torch.cat([prompt_embeds, cls_embed], dim=1)
        prompt_mask = torch.cat([prompt_mask, cls_mask], dim=1)

        prompt_embeds = ge.prompt_layer_norm(ge.final_proj(prompt_embeds))

        prompt_attention_mask = create_bidirectional_mask(
            config=ge.config, input_embeds=prompt_embeds, attention_mask=prompt_mask
        )

        for layer in ge.layers:
            prompt_embeds = layer(
                prompt_feats=prompt_embeds,
                vision_feats=vision_feats_flat,
                vision_pos_encoding=vision_pos_embeds_flat,
                prompt_mask=prompt_attention_mask,
            )

        return ge.output_layer_norm(prompt_embeds), prompt_mask

    @staticmethod
    def _box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
        x_c, y_c, w, h = x.unbind(-1)
        return torch.stack(
            [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=-1
        )


class DecoderWrapper(nn.Module):
    """DETR Encoder + Decoder + Mask Decoder"""

    def __init__(self, sam3_model: Sam3Model):
        super().__init__()
        self.detr_encoder = sam3_model.detr_encoder
        self.detr_decoder = sam3_model.detr_decoder
        self.mask_decoder = sam3_model.mask_decoder
        self.dot_product_scoring = sam3_model.dot_product_scoring
        self.box_head = sam3_model.detr_decoder.box_head

    def forward(
        self,
        fpn_feat_0,
        fpn_feat_1,
        fpn_feat_2,
        fpn_pos_2,
        prompt_features,
        prompt_mask,
    ):
        encoder_outputs = self.detr_encoder(
            vision_features=[fpn_feat_2],
            text_features=prompt_features,
            vision_pos_embeds=[fpn_pos_2],
            text_mask=prompt_mask,
        )

        decoder_outputs = self.detr_decoder(
            vision_features=encoder_outputs.last_hidden_state,
            text_features=encoder_outputs.text_features,
            vision_pos_encoding=encoder_outputs.pos_embeds_flattened,
            text_mask=prompt_mask,
            spatial_shapes=encoder_outputs.spatial_shapes,
        )

        all_box_offsets = self.box_head(decoder_outputs.intermediate_hidden_states)
        reference_boxes_inv_sig = self._inverse_sigmoid(decoder_outputs.reference_boxes)
        all_pred_boxes = self._box_cxcywh_to_xyxy(
            (reference_boxes_inv_sig + all_box_offsets).sigmoid()
        )

        all_pred_logits = self.dot_product_scoring(
            decoder_hidden_states=decoder_outputs.intermediate_hidden_states,
            text_features=encoder_outputs.text_features,
            text_mask=prompt_mask,
        ).squeeze(-1)

        pred_logits = all_pred_logits[-1]
        pred_boxes = all_pred_boxes[-1]
        decoder_hidden_states = decoder_outputs.intermediate_hidden_states[-1]
        presence_logits = decoder_outputs.presence_logits[-1]

        mask_outputs = self.mask_decoder(
            decoder_queries=decoder_hidden_states,
            backbone_features=[fpn_feat_0, fpn_feat_1, fpn_feat_2],
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            prompt_features=prompt_features,
            prompt_mask=prompt_mask,
        )

        return mask_outputs.pred_masks, pred_boxes, pred_logits, presence_logits

    @staticmethod
    def _inverse_sigmoid(x, eps=1e-3):
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))

    @staticmethod
    def _box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        return torch.stack(
            [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=-1
        )


def export_vision_encoder(model: Sam3Model, output_dir: Path, device: str = "cuda"):
    print("Exporting Vision Encoder...")
    wrapper = VisionEncoderWrapper(model, device=device).to(device).eval()

    torch.onnx.export(
        wrapper,
        (torch.randn(1, 3, 1008, 1008, device=device),),
        str(output_dir / "vision-encoder.onnx"),
        input_names=["images"],
        output_names=["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "images": {0: "batch"},
            "fpn_feat_0": {0: "batch"},
            "fpn_feat_1": {0: "batch"},
            "fpn_feat_2": {0: "batch"},
            "fpn_pos_2": {0: "batch"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'vision-encoder.onnx'}")


def export_text_encoder(model: Sam3Model, output_dir: Path, device: str = "cuda"):
    print("Exporting Text Encoder...")
    wrapper = TextEncoderWrapper(model).to(device).eval()

    torch.onnx.export(
        wrapper,
        (
            torch.randint(0, 49408, (1, 32), device=device),
            torch.ones(1, 32, dtype=torch.long, device=device),
        ),
        str(output_dir / "text-encoder.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_features", "text_mask"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "text_features": {0: "batch"},
            "text_mask": {0: "batch"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'text-encoder.onnx'}")


def export_geometry_encoder(model: Sam3Model, output_dir: Path, device: str = "cuda"):
    print("Exporting Geometry Encoder...")
    wrapper = GeometryEncoderWrapper(model).to(device).eval()

    torch.onnx.export(
        wrapper,
        (
            torch.rand(1, 5, 4, device=device),
            torch.ones(1, 5, dtype=torch.long, device=device),
            torch.randn(1, 256, 72, 72, device=device),
            torch.randn(1, 256, 72, 72, device=device),
        ),
        str(output_dir / "geometry-encoder.onnx"),
        input_names=["input_boxes", "input_boxes_labels", "fpn_feat_2", "fpn_pos_2"],
        output_names=["geometry_features", "geometry_mask"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "input_boxes": {0: "batch", 1: "num_boxes"},
            "input_boxes_labels": {0: "batch", 1: "num_boxes"},
            "fpn_feat_2": {0: "batch"},
            "fpn_pos_2": {0: "batch"},
            "geometry_features": {0: "batch", 1: "num_prompts"},
            "geometry_mask": {0: "batch", 1: "num_prompts"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'geometry-encoder.onnx'}")


def export_decoder(model: Sam3Model, output_dir: Path, device: str = "cuda"):
    print("Exporting Decoder...")
    wrapper = DecoderWrapper(model).to(device).eval()

    torch.onnx.export(
        wrapper,
        (
            torch.randn(1, 256, 288, 288, device=device),
            torch.randn(1, 256, 144, 144, device=device),
            torch.randn(1, 256, 72, 72, device=device),
            torch.randn(1, 256, 72, 72, device=device),
            torch.randn(1, 32, 256, device=device),
            torch.ones(1, 32, dtype=torch.bool, device=device),
        ),
        str(output_dir / "decoder.onnx"),
        input_names=[
            "fpn_feat_0",
            "fpn_feat_1",
            "fpn_feat_2",
            "fpn_pos_2",
            "prompt_features",
            "prompt_mask",
        ],
        output_names=["pred_masks", "pred_boxes", "pred_logits", "presence_logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            **{f"fpn_feat_{i}": {0: "batch"} for i in range(3)},
            "fpn_pos_2": {0: "batch"},
            "prompt_features": {0: "batch", 1: "prompt_len"},
            "prompt_mask": {0: "batch", 1: "prompt_len"},
            "pred_masks": {0: "batch"},
            "pred_boxes": {0: "batch"},
            "pred_logits": {0: "batch"},
            "presence_logits": {0: "batch"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'decoder.onnx'}")


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM3 model to ONNX format with dynamic batch support"
    )
    parser.add_argument(
        "--module", type=str, choices=["vision", "text", "geometry", "decoder"]
    )
    parser.add_argument("--all", action="store_true", help="Export all modules")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to SAM3 model directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="onnx-models", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not args.module and not args.all:
        parser.error("Please specify --module or --all")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM3 from {args.model_path}...")
    model = Sam3Model.from_pretrained(args.model_path).to(args.device).eval()
    print("  ✓ Model loaded\n")

    modules = ["vision", "text", "geometry", "decoder"] if args.all else [args.module]

    with torch.no_grad():
        for m in modules:
            {
                "vision": export_vision_encoder,
                "text": export_text_encoder,
                "geometry": export_geometry_encoder,
                "decoder": export_decoder,
            }[m](model, output_dir, args.device)

    print(f"\n✓ Export complete! Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
