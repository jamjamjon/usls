"""
SAM3 ONNX Inference Script
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

TARGET_SIZE = 1008


def parse_box_prompts(box_str: str) -> tuple[list, list]:
    """Parse box prompts string

    Format: "pos:x,y,w,h;neg:x,y,w,h;..." (xywh format)
    Returns: boxes [[x,y,w,h], ...], labels [1, 0, ...]
    """
    boxes, labels = [], []
    for part in box_str.split(";"):
        part = part.strip()
        if not part:
            continue
        if part.startswith("pos:"):
            label, coords = 1, part[4:]
        elif part.startswith("neg:"):
            label, coords = 0, part[4:]
        else:
            label, coords = 1, part  # default positive
        x, y, w, h = [float(v) for v in coords.split(",")]
        boxes.append([x, y, w, h])
        labels.append(label)
    return boxes, labels


def xywh_to_cxcywh_normalized(boxes: list, img_w: int, img_h: int) -> np.ndarray:
    """Convert xywh (pixel) to cxcywh (normalized)"""
    result = []
    for x, y, w, h in boxes:
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        result.append([cx, cy, nw, nh])
    return np.array(result, dtype=np.float32)


class Sam3ONNXInference:
    """SAM3 ONNX Inference Engine"""

    def __init__(
        self,
        vision_encoder_path: str,
        text_encoder_path: str,
        geometry_encoder_path: str,
        decoder_path: str,
        tokenizer_path: str,
        device: str = "cuda",
    ):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        print("Loading ONNX models...")
        self.vision_encoder = ort.InferenceSession(
            vision_encoder_path, providers=providers
        )
        self.text_encoder = ort.InferenceSession(text_encoder_path, providers=providers)
        self.geometry_encoder = ort.InferenceSession(
            geometry_encoder_path, providers=providers
        )
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(length=32, pad_id=49407)
        self.tokenizer.enable_truncation(max_length=32)
        print("  ✓ All models loaded")

    def preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Preprocess: resize to target size and normalize"""
        orig_size = image.shape[:2]  # (h, w)
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image)
        resized = np.array(
            pil_image.resize((TARGET_SIZE, TARGET_SIZE), PILImage.BILINEAR)
        )
        normalized = resized.astype(np.float32) / 127.5 - 1.0  # [0,255] -> [-1,1]
        tensor = normalized.transpose(2, 0, 1)[np.newaxis]  # NCHW
        return tensor, orig_size

    def encode_image(self, pixel_values: np.ndarray) -> dict:
        """Encode image using vision encoder"""
        outputs = self.vision_encoder.run(None, {"images": pixel_values})
        return {
            "fpn_feat_0": outputs[0],  # [B, 256, 288, 288]
            "fpn_feat_1": outputs[1],  # [B, 256, 144, 144]
            "fpn_feat_2": outputs[2],  # [B, 256, 72, 72]
            "fpn_pos_2": outputs[3],  # [B, 256, 72, 72]
        }

    def encode_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Encode text prompt"""
        self.tokenizer.enable_padding(pad_id=49407, length=32)
        self.tokenizer.enable_truncation(max_length=32)

        encoded = self.tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        outputs = self.text_encoder.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        return outputs[0], outputs[1]

    def encode_boxes(
        self,
        boxes: np.ndarray,
        labels: np.ndarray,
        fpn_feat: np.ndarray,
        fpn_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode box prompts"""
        outputs = self.geometry_encoder.run(
            None,
            {
                "input_boxes": boxes.astype(np.float32),
                "input_boxes_labels": labels.astype(np.int64),
                "fpn_feat_2": fpn_feat,
                "fpn_pos_2": fpn_pos,
            },
        )
        return outputs[0], outputs[1]

    def decode(
        self,
        vision_features: dict,
        prompt_features: np.ndarray,
        prompt_mask: np.ndarray,
    ) -> dict:
        """Decode features to generate masks"""
        outputs = self.decoder.run(
            None,
            {
                "fpn_feat_0": vision_features["fpn_feat_0"],
                "fpn_feat_1": vision_features["fpn_feat_1"],
                "fpn_feat_2": vision_features["fpn_feat_2"],
                "fpn_pos_2": vision_features["fpn_pos_2"],
                "prompt_features": prompt_features,
                "prompt_mask": prompt_mask,
            },
        )
        return {
            "pred_masks": outputs[0],
            "pred_boxes": outputs[1],
            "pred_logits": outputs[2],
            "presence_logits": outputs[3],
        }

    def predict(
        self,
        image: np.ndarray,
        text: Optional[str] = None,
        boxes: Optional[list] = None,
        box_labels: Optional[list] = None,
        conf_threshold: float = 0.3,
    ) -> dict:
        """Unified prediction with text and/or box prompts

        Args:
            image: RGB image [H, W, 3]
            text: Text prompt (optional)
            boxes: Box prompts [[x,y,w,h], ...] in xywh pixel format (optional)
            box_labels: Box labels [1, 0, ...] 1=pos, 0=neg (optional)
            conf_threshold: Confidence threshold
        """
        pixel_values, orig_size = self.preprocess_image(image)
        vision_features = self.encode_image(pixel_values)
        h, w = orig_size

        # Encode text
        if text:
            text_features, text_mask = self.encode_text(text)
        else:
            # No text: use padding tokens (length=32)
            pad_ids = np.full((1, 32), 49407, dtype=np.int64)
            pad_mask = np.zeros((1, 32), dtype=np.int64)
            pad_mask[0, 0] = 1  # at least one valid token
            outputs = self.text_encoder.run(
                None, {"input_ids": pad_ids, "attention_mask": pad_mask}
            )
            text_features, text_mask = outputs[0], outputs[1]

        # Encode boxes
        if boxes and len(boxes) > 0:
            boxes_cxcywh = xywh_to_cxcywh_normalized(boxes, w, h)
            boxes_array = boxes_cxcywh.reshape(1, -1, 4)
            if box_labels:
                labels_array = np.array(box_labels, dtype=np.int64).reshape(1, -1)
            else:
                labels_array = np.ones((1, len(boxes)), dtype=np.int64)
            geom_features, geom_mask = self.encode_boxes(
                boxes_array,
                labels_array,
                vision_features["fpn_feat_2"],
                vision_features["fpn_pos_2"],
            )
            # Concatenate text and geometry features
            prompt_features = np.concatenate([text_features, geom_features], axis=1)
            prompt_mask = np.concatenate([text_mask, geom_mask], axis=1)
        else:
            # No boxes: use text features only
            prompt_features = text_features
            prompt_mask = text_mask

        outputs = self.decode(vision_features, prompt_features, prompt_mask)
        return self._postprocess(outputs, orig_size, conf_threshold, boxes)

    def _postprocess(
        self,
        outputs: dict,
        orig_size: tuple[int, int],
        conf_threshold: float,
        input_boxes: Optional[list] = None,
    ) -> dict:
        """Post-process model outputs"""
        pred_masks = outputs["pred_masks"][0]
        pred_boxes = outputs["pred_boxes"][0]
        pred_logits = outputs["pred_logits"][0]
        presence_logits = outputs["presence_logits"][0, 0]

        presence_score = 1 / (1 + np.exp(-presence_logits))
        scores = (1 / (1 + np.exp(-pred_logits))) * presence_score
        keep = scores > conf_threshold

        h, w = orig_size
        # Resize masks: 288x288 -> original size
        masks = []
        for m in pred_masks[keep]:
            mask_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            masks.append(mask_resized > 0)
        # Scale boxes from normalized [0,1] to pixel coordinates
        boxes = pred_boxes[keep].copy()
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        boxes = np.clip(boxes, 0, [[w, h, w, h]])

        return {
            "masks": masks,
            "boxes": boxes,
            "scores": scores[keep],
            "orig_size": orig_size,
            "input_boxes": input_boxes,
        }


def visualize_results(
    image: np.ndarray, results: dict, output_path: str, alpha: float = 0.35
):
    """Visualize detection results with mask overlay and contours"""
    vis = image.copy()
    colors = [
        (30, 144, 255),  # Dodger Blue
        (255, 144, 30),  # Orange
        (144, 255, 30),  # Green-Yellow
        (255, 30, 144),  # Pink
        (30, 255, 144),  # Spring Green
    ]

    for i, (mask, box, score) in enumerate(
        zip(results["masks"], results["boxes"], results["scores"])
    ):
        color = colors[i % len(colors)]
        mask_bool = mask > 0

        # Apply mask overlay (lighter)
        overlay = vis.copy()
        overlay[mask_bool] = color
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)

        # Draw mask contours
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 2)

        # Draw box and score
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis, f"{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    cv2.imwrite(output_path, vis)
    print(f"  ✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 ONNX Inference")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--text", type=str, help="Text prompt")
    parser.add_argument(
        "--boxes", type=str, help="Box prompts: pos:x,y,w,h;neg:x,y,w,h (xywh format)"
    )
    parser.add_argument(
        "--model-dir", type=str, default="onnx-models", help="ONNX models directory"
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Path to tokenizer.json"
    )
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not args.text and not args.boxes:
        parser.error("Please specify --text or --boxes")

    # Load model
    model_dir = Path(args.model_dir)
    engine = Sam3ONNXInference(
        vision_encoder_path=str(model_dir / "vision-encoder.onnx"),
        text_encoder_path=str(model_dir / "text-encoder.onnx"),
        geometry_encoder_path=str(model_dir / "geometry-encoder.onnx"),
        decoder_path=str(model_dir / "decoder.onnx"),
        tokenizer_path=args.tokenizer,
        device=args.device,
    )

    # Load image
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise ValueError(f"Cannot load image: {args.image}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"\nProcessing: {args.image} ({image.shape[1]}x{image.shape[0]})")

    # Parse prompts
    boxes, box_labels = None, None
    if args.boxes:
        boxes, box_labels = parse_box_prompts(args.boxes)
        print(f"  Box prompts: {len(boxes)} boxes, labels={box_labels}")
    if args.text:
        print(f"  Text prompt: '{args.text}'")

    # Run inference
    results = engine.predict(
        image,
        text=args.text,
        boxes=boxes,
        box_labels=box_labels,
        conf_threshold=args.conf,
    )

    print(f"  Found {len(results['masks'])} objects")

    # Visualize
    visualize_results(image_bgr, results, args.output)


if __name__ == "__main__":
    main()
