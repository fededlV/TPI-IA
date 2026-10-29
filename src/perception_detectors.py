"""Open-vocabulary perception: OWLv2 baseline detector (EXP-01) and model-free post-processing helpers.

The detector receives only the image and the 120-category vocabulary; it never reads annotations.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.lvis_evaluator import box_iou

OWLV2_BASE = "google/owlv2-base-patch16-ensemble"
PROMPT_TEMPLATE = "a photo of a {}"
DEFAULT_MIN_SCORE = 0.05  # raw cache floor; the operating threshold is tuned on CAL afterwards
NMS_IOU = 0.5
MAX_DETECTIONS = 300


@dataclass(frozen=True)
class Detection:
    category_id: int
    score: float
    bbox: tuple[float, float, float, float]  # xyxy, absolute pixels inside the image
    source: str = "owlv2"


def clip_box(box, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Clip an xyxy box to the image; None if nothing of it remains inside."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = min(max(x1, 0.0), float(width)), min(max(x2, 0.0), float(width))
    y1, y2 = min(max(y1, 0.0), float(height)), min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def per_class_nms(detections: list[Detection], iou_threshold: float = NMS_IOU) -> list[Detection]:
    """Greedy NMS within each category; output ranked by descending score."""
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: (-d.score, d.category_id, d.bbox)):
        if all(k.category_id != det.category_id or box_iou(list(k.bbox), list(det.bbox)) < iou_threshold
               for k in kept):
            kept.append(det)
    return kept


def filter_detections(detections: list[Detection], score_threshold: float,
                      max_detections: int = MAX_DETECTIONS) -> list[Detection]:
    ranked = sorted((d for d in detections if d.score >= score_threshold), key=lambda d: -d.score)
    return ranked[:max_detections]


class OwlV2Detector:
    """OWLv2 text-prompted detector over the official category names (aliases are reserved for EXP-02A)."""

    def __init__(self, categories: dict[int, str], model_id: str = OWLV2_BASE, device: str = "cpu",
                 min_score: float = DEFAULT_MIN_SCORE, nms_iou: float = NMS_IOU):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.name = f"owlv2:{model_id}"
        self.model_id, self.device, self.min_score, self.nms_iou = model_id, device, min_score, nms_iou
        self.category_ids = sorted(categories)
        self.prompts = [PROMPT_TEMPLATE.format(categories[c]) for c in self.category_ids]
        self._torch = torch
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()

    def detect(self, image) -> list[Detection]:
        """Raw detections (score >= min_score) after per-class NMS, clipped to the image."""
        image = image.convert("RGB")
        width, height = image.size
        inputs = self.processor(text=[self.prompts], images=image, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        side = max(width, height)  # OWLv2 pads to a square (bottom/right) before resizing
        result = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=self.min_score, target_sizes=[(side, side)])[0]
        detections = []
        for score, label, box in zip(result["scores"].tolist(), result["labels"].tolist(), result["boxes"].tolist()):
            clipped = clip_box(box, width, height)
            if clipped is not None:
                detections.append(Detection(self.category_ids[label], float(score), clipped))
        return per_class_nms(detections, self.nms_iou)
