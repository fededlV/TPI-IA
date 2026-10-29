"""End-to-end inference: image -> detections -> masks -> SceneJSON, traced per stage, plus detection cache I/O.

`infer_scene` only receives an image path and injected models, so it works on arbitrary external images
and never touches development annotations.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src import scene_builder as sb
from src.perception_detectors import Detection, filter_detections
from src.trace_logger import ExecutionTracer


def build_scene(image_id: str, image: Image.Image, detections: list[Detection], masks: list[tuple[np.ndarray, float]],
                categories: dict[int, str], tracer: ExecutionTracer | None = None) -> tuple[dict, int]:
    """SceneJSON from already-thresholded detections and their masks; returns (scene, n_mask_fallbacks)."""
    rgb = np.asarray(image.convert("RGB"))
    width, height = image.size
    objects = []
    for i, (det, (mask, sam_score)) in enumerate(zip(detections, masks)):
        obj_id = f"obj_{i:03d}"
        objects.append(sb.build_object(obj_id, det.category_id, det.score, det.bbox, mask, image_rgb=rgb,
                                       category_name=categories.get(det.category_id), detector_source=det.source))
        if tracer is not None:
            tracer.trace_object(image_id, obj_id, detector_source=det.source, raw_score=det.score,
                                calibrated_score=det.score, sam2_candidate_idx=0, sam2_iou_score=sam_score,
                                mask_fallback="mask_fallback" in objects[-1])
    n_fallback = sum("mask_fallback" in o for o in objects)
    return sb.build_scene_json(image_id, width, height, objects), n_fallback


def segment_and_build(image_id: str, image: Image.Image, detections: list[Detection], segmenter,
                      categories: dict[int, str], tracer: ExecutionTracer | None = None) -> tuple[dict, int]:
    tracer = tracer or ExecutionTracer()
    with tracer.stage(image_id, "segmentation", {"model": segmenter.name, "multimask_output": False}) as rec:
        masks = segmenter.segment(image, [d.bbox for d in detections])
        rec.set_counts(len(detections), len(masks))
    with tracer.stage(image_id, "scene_build", {}) as rec:
        scene, n_fallback = build_scene(image_id, image, detections, masks, categories, tracer)
        rec.set_counts(len(detections), len(scene["objects"]))
        if n_fallback:
            rec.warn(f"{n_fallback} empty SAM2 mask(s) replaced by bbox fill")
    return scene, n_fallback


def infer_scene(image_path, detector, segmenter, categories: dict[int, str], score_threshold: float,
                tracer: ExecutionTracer | None = None) -> tuple[dict, dict]:
    """Full blind inference on one image file; returns (SceneJSON, run info)."""
    image_path = Path(image_path)
    image_id = image_path.stem
    tracer = tracer or ExecutionTracer()
    image = Image.open(image_path).convert("RGB")
    with tracer.stage(image_id, "detection", {"model": detector.name, "score_threshold": score_threshold}) as rec:
        raw = detector.detect(image)
        kept = filter_detections(raw, score_threshold)
        rec.set_counts(len(raw), len(kept))
    scene, n_fallback = segment_and_build(image_id, image, kept, segmenter, categories, tracer)
    return scene, {"n_detections_raw": len(raw), "n_kept": len(kept), "n_mask_fallbacks": n_fallback}


# --------------------------------------------------------------------------- detection cache


def save_detection_cache(detections_by_image: dict[str, list[Detection]], path, meta: dict | None = None,
                         ndigits: int | None = None) -> Path:
    """JSON detection cache; a '.gz' suffix gzips it and `ndigits` rounds scores/coordinates to keep it small."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    r = (lambda v: round(v, ndigits)) if ndigits is not None else (lambda v: v)
    payload = {"meta": meta or {}, "detections": {
        key: [{"category_id": d.category_id, "score": r(d.score), "bbox": [r(v) for v in d.bbox], "source": d.source}
              for d in dets]
        for key, dets in detections_by_image.items()}}
    text = json.dumps(payload)
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
    else:
        path.write_text(text, encoding="utf-8")
    return path


def load_detection_cache(path) -> tuple[dict[str, list[Detection]], dict]:
    path = Path(path)
    text = gzip.decompress(path.read_bytes()).decode("utf-8") if path.suffix == ".gz" else path.read_text(
        encoding="utf-8")
    payload = json.loads(text)
    detections = {key: [Detection(d["category_id"], d["score"], tuple(d["bbox"]), d["source"]) for d in dets]
                  for key, dets in payload["detections"].items()}
    return detections, payload["meta"]
