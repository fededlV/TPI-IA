"""Box-prompted SAM2.1 segmentation (EXP-01 baseline: multimask_output=False) with a non-empty mask guarantee."""

from __future__ import annotations

import numpy as np

from src import scene_builder as sb
from src.config_and_seeds import resolve_device

SAM2_BASE_PLUS = "facebook/sam2.1-hiera-base-plus"
BOX_CHUNK = 32


def guarantee_non_empty(mask: np.ndarray, box) -> tuple[np.ndarray, bool]:
    """sum(mask) >= 1 always (schema area_ratio > 0): fall back to the box fill, flagged for lineage."""
    return sb.ensure_non_empty_mask(mask, box)


class Sam2Segmenter:
    """One image encoding per image, then box prompts decoded in chunks with a single mask per box."""

    def __init__(self, model_id: str = SAM2_BASE_PLUS, device: str | None = None, box_chunk: int = BOX_CHUNK):
        import torch
        from transformers import Sam2Model, Sam2Processor

        device = device or resolve_device()
        self.name = f"sam2:{model_id}"
        self.model_id, self.device, self.box_chunk = model_id, device, box_chunk
        self._torch = torch
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(device).eval()

    def segment(self, image, boxes: list) -> list[tuple[np.ndarray, float]]:
        """Return (binary HxW mask, predicted IoU score) per xyxy box, in input order (masks may be empty)."""
        if not boxes:
            return []
        image = image.convert("RGB")
        torch = self._torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        original_sizes = inputs["original_sizes"]
        results: list[tuple[np.ndarray, float]] = []
        with torch.inference_mode():
            embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
            for start in range(0, len(boxes), self.box_chunk):
                chunk = [[float(v) for v in b] for b in boxes[start:start + self.box_chunk]]
                prompt = self.processor(input_boxes=[chunk], original_sizes=original_sizes,
                                        return_tensors="pt").to(self.device)
                outputs = self.model(image_embeddings=embeddings, input_boxes=prompt["input_boxes"],
                                     multimask_output=False)
                masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), original_sizes.cpu())[0]
                scores = outputs.iou_scores.cpu().reshape(len(chunk), -1)[:, 0].tolist()
                for k in range(len(chunk)):
                    results.append((masks[k, 0].numpy().astype(bool), float(scores[k])))
        return results
