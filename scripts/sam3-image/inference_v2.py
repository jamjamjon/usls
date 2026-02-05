import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def parse_box_prompts(box_str: str) -> tuple[list, list]:
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
            label, coords = 1, part
        x, y, w, h = [float(v) for v in coords.split(",")]
        boxes.append([x, y, w, h])
        labels.append(label)
    return boxes, labels


def xywh_to_cxcywh_normalized(boxes: list, img_w: int, img_h: int) -> np.ndarray:
    result = []
    for x, y, w, h in boxes:
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        result.append([cx, cy, nw, nh])
    return np.array(result, dtype=np.float32)


def xywh_to_xyxy(boxes: list) -> np.ndarray:
    arr = np.array(boxes, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 4)
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 0] + arr[:, 2]
    y2 = arr[:, 1] + arr[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


class Sam3ONNXInferenceV2:
    def __init__(
        self,
        vision_encoder_path: str,
        text_encoder_path: str,
        decoder_path: str,
        tokenizer_path: str,
        image_height: int = 504,
        image_width: int = 896,
        device: str = "cuda",
    ):
        self.image_height = image_height
        self.image_width = image_width

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        self.vision_encoder = ort.InferenceSession(
            vision_encoder_path, providers=providers
        )
        self.text_encoder = ort.InferenceSession(text_encoder_path, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(length=32, pad_id=49407)
        self.tokenizer.enable_truncation(max_length=32)

    def preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        orig_size = image.shape[:2]
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image)
        resized = np.array(
            pil_image.resize((self.image_width, self.image_height), PILImage.BILINEAR)
        )
        normalized = resized.astype(np.float32) / 127.5 - 1.0
        tensor = normalized.transpose(2, 0, 1)[np.newaxis]
        return tensor, orig_size

    def encode_image(self, pixel_values: np.ndarray) -> dict:
        outputs = self.vision_encoder.run(None, {"images": pixel_values})
        return {
            "fpn_feat_0": outputs[0],
            "fpn_feat_1": outputs[1],
            "fpn_feat_2": outputs[2],
            "fpn_pos_2": outputs[3],
        }

    def encode_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        encoded = self.tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        outputs = self.text_encoder.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        return outputs[0], outputs[1]

    def decode(
        self,
        vision_features: dict,
        text_features: np.ndarray,
        text_mask: np.ndarray,
        input_boxes: np.ndarray,
        input_boxes_labels: np.ndarray,
    ) -> dict:
        outputs = self.decoder.run(
            None,
            {
                "fpn_feat_0": vision_features["fpn_feat_0"],
                "fpn_feat_1": vision_features["fpn_feat_1"],
                "fpn_feat_2": vision_features["fpn_feat_2"],
                "fpn_pos_2": vision_features["fpn_pos_2"],
                "text_features": text_features,
                "text_mask": text_mask,
                "input_boxes": input_boxes,
                "input_boxes_labels": input_boxes_labels,
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
        # suppress_neg: bool = False,
        # suppress_neg_iou: float = 0.3,
    ) -> dict:
        pixel_values, orig_size = self.preprocess_image(image)
        vision_features = self.encode_image(pixel_values)

        if text:
            text_features, text_mask = self.encode_text(text)
        else:
            pad_ids = np.full((1, 32), 49407, dtype=np.int64)
            pad_mask = np.zeros((1, 32), dtype=np.int64)
            pad_mask[0, 0] = 1
            outputs = self.text_encoder.run(
                None, {"input_ids": pad_ids, "attention_mask": pad_mask}
            )
            text_features, text_mask = outputs[0], outputs[1]

        h, w = orig_size
        if boxes and len(boxes) > 0:
            sx = float(self.image_width) / float(w)
            sy = float(self.image_height) / float(h)
            boxes_resized = [
                [x * sx, y * sy, bw * sx, bh * sy] for x, y, bw, bh in boxes
            ]
            boxes_cxcywh = xywh_to_cxcywh_normalized(
                boxes_resized, self.image_width, self.image_height
            )
            input_boxes = boxes_cxcywh.reshape(1, -1, 4).astype(np.float32)
            if box_labels:
                input_boxes_labels = np.array(box_labels, dtype=np.int64).reshape(1, -1)
            else:
                input_boxes_labels = np.ones((1, input_boxes.shape[1]), dtype=np.int64)
        else:
            input_boxes = np.zeros((1, 1, 4), dtype=np.float32)
            input_boxes_labels = np.full((1, 1), -10, dtype=np.int64)

        outputs = self.decode(
            vision_features,
            text_features=text_features,
            text_mask=text_mask,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
        )
        return self._postprocess(
            outputs,
            orig_size,
            conf_threshold,
            input_boxes_xywh=boxes,
            input_boxes_labels=box_labels,
            # suppress_neg=suppress_neg,
            # suppress_neg_iou=suppress_neg_iou,
        )

    def _postprocess(
        self,
        outputs: dict,
        orig_size: tuple[int, int],
        conf_threshold: float,
        input_boxes_xywh: Optional[list] = None,
        input_boxes_labels: Optional[list] = None,
        suppress_neg: bool = False,
        suppress_neg_iou: float = 0.3,
    ) -> dict:
        pred_masks = outputs["pred_masks"][0]
        pred_boxes = outputs["pred_boxes"][0]
        pred_logits = outputs["pred_logits"][0]
        presence_logits = outputs["presence_logits"][0, 0]

        presence_score = 1 / (1 + np.exp(-presence_logits))
        scores = (1 / (1 + np.exp(-pred_logits))) * presence_score
        keep = scores > conf_threshold

        h, w = orig_size
        masks = []
        for m in pred_masks[keep]:
            mask_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            masks.append(mask_resized > 0)

        boxes = pred_boxes[keep].copy()
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        boxes = np.clip(boxes, 0, [[w, h, w, h]])
        scores = scores[keep]

        return {
            "masks": masks,
            "boxes": boxes,
            "scores": scores,
            "orig_size": orig_size,
        }


def visualize_results(
    image: np.ndarray, results: dict, output_path: str, alpha: float = 0.35
):
    vis = image.copy()
    colors = [
        (30, 144, 255),
        (255, 144, 30),
        (144, 255, 30),
        (255, 30, 144),
        (30, 255, 144),
    ]

    for i, (mask, box, score) in enumerate(
        zip(results["masks"], results["boxes"], results["scores"])
    ):
        color = colors[i % len(colors)]
        mask_bool = mask > 0

        overlay = vis.copy()
        overlay[mask_bool] = color
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)

        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 2)

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"{score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.imwrite(output_path, vis)


def main():
    parser = argparse.ArgumentParser(description="SAM3 ONNX Inference (v2)")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--text", type=str, help="Text prompt")
    parser.add_argument(
        "--boxes", type=str, help="Box prompts: pos:x,y,w,h;neg:x,y,w,h (xywh format)"
    )
    parser.add_argument(
        "--model-dir", type=str, default="onnx-models-v2", help="ONNX models directory"
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Path to tokenizer.json"
    )
    parser.add_argument(
        "--output", type=str, default="output-v2.png", help="Output path"
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-height", type=int, default=1008)
    parser.add_argument("--image-width", type=int, default=1008)
    args = parser.parse_args()

    if not args.text and not args.boxes:
        parser.error("Please specify --text or --boxes")

    model_dir = Path(args.model_dir)
    engine = Sam3ONNXInferenceV2(
        vision_encoder_path=str(model_dir / "vision-encoder-q4f16.onnx"),
        text_encoder_path=str(model_dir / "text-encoder-q4f16.onnx"),
        decoder_path=str(model_dir / "geo-encoder-mask-decoder-q4f16.onnx"),
        tokenizer_path=args.tokenizer,
        image_height=args.image_height,
        image_width=args.image_width,
        device=args.device,
    )

    for img in Path(args.image).glob("*.jpg"):
        print(img)
        image_bgr = cv2.imread(str(img))
        if image_bgr is None:
            raise ValueError(f"Cannot load image: {img}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        boxes, box_labels = None, None
        if args.boxes:
            boxes, box_labels = parse_box_prompts(args.boxes)

        # start = time.time()
        results = engine.predict(
            image,
            text=args.text,
            boxes=boxes,
            box_labels=box_labels,
            conf_threshold=args.conf,
        )
        # print(f"Time: {(time.time() - start) * 1000} ms")

        visualize_results(image_bgr, results, args.output)


if __name__ == "__main__":
    main()
