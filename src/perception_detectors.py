"""Open-vocabulary perception: OWLv2 detector (EXP-01 baseline, EXP-02A chunked prompting with LVIS aliases),
EXP-02B 2x2 overlapping tiles, and model-free post-processing helpers.

The detector receives only the image and the 120-category vocabulary; it never reads annotations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from src.config_and_seeds import build_prompt_aliases, resolve_device
from src.lvis_evaluator import box_iou

OWLV2_BASE = "google/owlv2-base-patch16-ensemble"
OWLV2_LARGE = "google/owlv2-large-patch14-ensemble"
PROMPT_TEMPLATE = "a photo of a {}"
DEFAULT_MIN_SCORE = 0.05  # raw cache floor; the operating threshold is tuned on CAL afterwards
NMS_IOU = 0.5
MAX_DETECTIONS = 300
SIBLING_IOU = 0.7
TILE_OVERLAP = 0.25  # shared band = 25% of the tile side
TILE_EDGE_MARGIN = 3.0  # px; boxes this close to an inner tile border are truncation artifacts

# EXP-02A: 5 semantic blocks x 24 classes. Every sibling group lives inside one block, so the per-block argmax
# arbitrates between look-alike classes on the same box.
SEMANTIC_BLOCKS: dict[str, tuple[str, ...]] = {
    "apparel": ("person", "jacket", "shirt", "jersey", "dress", "trousers", "jean", "short pants", "sock", "shoe",
                "boot", "ski boot", "glove", "baseball glove", "belt", "necktie", "hat", "baseball cap", "helmet",
                "button", "strap", "tag", "backpack", "handbag"),
    "accessories_electronics": ("necklace", "earring", "bracelet", "ring", "watch", "clock", "spectacles",
                                "sunglasses", "goggles", "cell phone", "laptop", "computer keyboard", "computer mouse",
                                "computer monitor", "television set", "book", "suitcase", "plastic bag", "box",
                                "basket", "teddy bear", "umbrella", "lamp", "mirror"),
    "kitchen_tableware": ("pizza", "banana", "broccoli", "bottle", "drinking glass", "cup", "bowl", "plate", "fork",
                          "knife", "napkin", "tablecloth", "vase", "flower arrangement", "flowerpot", "table",
                          "refrigerator", "sink", "faucet", "cabinet", "drawer", "handle", "doorknob", "knob"),
    "home_street_fixtures": ("sofa", "chair", "pillow", "blanket", "bed", "curtain", "toilet", "wall socket", "vent",
                             "trash can", "bench", "awning", "banner", "flag", "signboard", "street sign",
                             "traffic light", "streetlight", "pole", "fireplug", "license plate", "headlight",
                             "taillight", "windshield wiper"),
    "vehicles_animals_sports": ("car", "bus", "train", "airplane", "boat", "motorcycle", "bicycle", "wheel", "horse",
                                "dog", "cat", "bird", "elephant", "giraffe", "zebra", "cow", "tennis racket",
                                "baseball bat", "ski", "ski pole", "skateboard", "surfboard", "kite", "frisbee"),
}

# Look-alike classes that must not both label the same object.
SIBLING_GROUPS: tuple[tuple[str, ...], ...] = (
    ("clock", "watch"), ("trousers", "jean", "short pants"), ("shoe", "boot", "ski boot"),
    ("handle", "doorknob", "knob"), ("spectacles", "sunglasses", "goggles"))

# EXP-02A factorial: {one prompt of 120 classes, 5 semantic blocks} x {official names, official names + aliases}.
VARIANTS = ("single_raw", "single_alias", "chunked_raw", "chunked_alias")


@dataclass(frozen=True)
class Detection:
    category_id: int
    score: float
    bbox: tuple[float, float, float, float]  # xyxy, absolute pixels inside the image
    source: str = "owlv2"


@dataclass(frozen=True)
class QueryTable:
    """Templated text queries and the official category each one is remapped to."""
    prompts: tuple[str, ...]
    category_ids: tuple[int, ...]
    is_official: tuple[bool, ...]


def clip_box(box, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Clip an xyxy box to the image; None if nothing of it remains inside."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = min(max(x1, 0.0), float(width)), min(max(x2, 0.0), float(width))
    y1, y2 = min(max(y1, 0.0), float(height)), min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    iw = np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])
    ih = np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1])
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def per_class_nms(detections: list[Detection], iou_threshold: float = NMS_IOU) -> list[Detection]:
    """Greedy NMS within each category; output ranked by descending score."""
    order_key = lambda d: (-d.score, d.category_id, d.bbox)  # noqa: E731
    by_class: dict[int, list[Detection]] = {}
    for det in sorted(detections, key=order_key):
        by_class.setdefault(det.category_id, []).append(det)
    kept: list[Detection] = []
    for dets in by_class.values():
        boxes = np.array([d.bbox for d in dets], dtype=float)
        alive = np.ones(len(dets), dtype=bool)
        for i in range(len(dets)):
            if alive[i]:
                kept.append(dets[i])
                alive[i + 1:] &= _iou_one_to_many(boxes[i], boxes[i + 1:]) < iou_threshold
    return sorted(kept, key=order_key)


def filter_detections(detections: list[Detection], score_threshold: float,
                      max_detections: int = MAX_DETECTIONS) -> list[Detection]:
    ranked = sorted((d for d in detections if d.score >= score_threshold), key=lambda d: -d.score)
    return ranked[:max_detections]


# --------------------------------------------------------------------------- EXP-02A: chunked prompting + aliases


def _ids_by_name(categories: dict[int, str], names, what: str) -> list[int]:
    by_name = {name: cid for cid, name in categories.items()}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"{what} references unknown categories: {unknown}")
    return [by_name[n] for n in names]


def build_blocks(categories: dict[int, str],
                 blocks: dict[str, tuple[str, ...]] | None = None) -> dict[str, list[int]]:
    """Semantic blocks as category ids; fails fast on unknown names."""
    blocks = SEMANTIC_BLOCKS if blocks is None else blocks
    return {name: _ids_by_name(categories, members, f"block {name!r}") for name, members in blocks.items()}


def sibling_groups(categories: dict[int, str]) -> list[frozenset[int]]:
    return [frozenset(_ids_by_name(categories, group, "sibling group")) for group in SIBLING_GROUPS]


def build_query_table(aliases: dict[int, dict], template: str = PROMPT_TEMPLATE) -> QueryTable:
    """Flatten LVIS_PROMPT_ALIASES into one query list; every alias query remaps to its official category_id."""
    prompts, category_ids, is_official = [], [], []
    for cid, entry in sorted(aliases.items()):
        for k, text in enumerate(entry["prompts"]):
            prompts.append(template.format(text))
            category_ids.append(cid)
            is_official.append(k == 0)
    if len(set(prompts)) != len(prompts):
        raise ValueError("Duplicate text queries: every prompt must map to exactly one category")
    return QueryTable(tuple(prompts), tuple(category_ids), tuple(is_official))


def category_scores(query_scores: np.ndarray, table: QueryTable, category_ids: list[int],
                    use_aliases: bool) -> np.ndarray:
    """[boxes, queries] sigmoid scores -> [boxes, categories]: official-name column, or max over all alias columns."""
    cols = np.asarray(table.category_ids)
    usable = np.ones(len(cols), dtype=bool) if use_aliases else np.asarray(table.is_official)
    out = np.zeros((query_scores.shape[0], len(category_ids)), dtype=query_scores.dtype)
    for j, cid in enumerate(category_ids):
        out[:, j] = query_scores[:, (cols == cid) & usable].max(axis=1)
    return out


def label_boxes(cat_scores: np.ndarray, boxes: np.ndarray, category_ids: list[int], blocks: list[list[int]],
                min_score: float, width: int, height: int, nms_iou: float = NMS_IOU,
                source: str = "owlv2") -> list[Detection]:
    """One label per box and per block (argmax within the block, kept if score > min_score as in HF OWLv2),
    clipped to the image, then per-class NMS. A single block of all categories reproduces EXP-01 semantics."""
    col = {cid: j for j, cid in enumerate(category_ids)}
    detections = []
    for block in blocks:
        idx = np.array([col[cid] for cid in block])
        sub = cat_scores[:, idx]
        best = sub.argmax(axis=1)
        best_score = sub[np.arange(len(sub)), best]
        for b in np.nonzero(best_score > min_score)[0]:
            clipped = clip_box(boxes[b], width, height)
            if clipped is not None:
                detections.append(Detection(int(block[best[b]]), float(best_score[b]), clipped, source))
    return per_class_nms(detections, nms_iou)


def extract_variants(query_scores: np.ndarray, boxes: np.ndarray, table: QueryTable, category_ids: list[int],
                     blocks: dict[str, list[int]], min_score: float, width: int, height: int,
                     nms_iou: float = NMS_IOU, source: str = "owlv2") -> dict[str, list[Detection]]:
    """The 4 EXP-02A variants from one forward pass (OWLv2 scores each query independently, so splitting the
    queries into blocks only changes the per-box argmax, never the scores)."""
    out = {}
    for use_aliases in (False, True):
        cat = category_scores(query_scores, table, category_ids, use_aliases)
        for layout, block_list in (("single", [list(category_ids)]), ("chunked", list(blocks.values()))):
            out[f"{layout}_{'alias' if use_aliases else 'raw'}"] = label_boxes(
                cat, boxes, category_ids, block_list, min_score, width, height, nms_iou, source)
    return out


def suppress_siblings(detections: list[Detection], groups: list[frozenset[int]],
                      iou_threshold: float = SIBLING_IOU) -> list[Detection]:
    """Cross-class NMS restricted to sibling groups: of two look-alike labels on the same object (IoU >= threshold),
    only the higher-scoring survives. Same-class duplicates are left to per_class_nms."""
    group_of = {cid: g for g in groups for cid in g}
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: (-d.score, d.category_id, d.bbox)):
        group = group_of.get(det.category_id)
        if group is None or all(k.category_id == det.category_id or k.category_id not in group
                                or box_iou(list(k.bbox), list(det.bbox)) < iou_threshold for k in kept):
            kept.append(det)
    return kept


# --------------------------------------------------------------------------- EXP-02B: 2x2 overlapping tiles


def tile_grid(width: int, height: int, overlap: float = TILE_OVERLAP) -> list[tuple[int, int, int, int]]:
    """Four xyxy windows (row-major) whose shared band is `overlap` of the tile side: tile = dim / (2 - overlap)."""
    tw, th = math.ceil(width / (2 - overlap)), math.ceil(height / (2 - overlap))
    return [(x, y, x + tw, y + th) for y in (0, height - th) for x in (0, width - tw)]


def project_box(box, tile) -> tuple[float, float, float, float]:
    x0, y0 = tile[0], tile[1]
    return (box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0)


def touches_inner_edge(box, tile, width: int, height: int, margin: float = TILE_EDGE_MARGIN) -> bool:
    """True if a global-frame box reaches a tile border that lies inside the image (a truncation artifact).
    Borders shared with the image border never count: objects there are cut by the photo, not by the tiling."""
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = tile
    return ((tx1 > 0 and x1 <= tx1 + margin) or (ty1 > 0 and y1 <= ty1 + margin)
            or (tx2 < width and x2 >= tx2 - margin) or (ty2 < height and y2 >= ty2 - margin))


def view_index(source: str) -> int | None:
    """Tile index encoded in a detection source ('owlv2:tile2' -> 2); None for the global view."""
    _, sep, idx = source.rpartition(":tile")
    return int(idx) if sep else None


def detect_with_tiles(image, detect_fn, base_source: str = "owlv2",
                      overlap: float = TILE_OVERLAP) -> dict[str, list[Detection]]:
    """Global pass + 4 tile passes of `detect_fn(image) -> {variant: detections}`. Tile boxes are projected to the
    global frame [0, W] x [0, H] and tagged '<base>:tile<i>'; truncation filtering is left to merge_views."""
    image = image.convert("RGB")
    width, height = image.size
    out = {variant: [replace(d, source=f"{base_source}:global") for d in dets]
           for variant, dets in detect_fn(image).items()}
    for i, tile in enumerate(tile_grid(width, height, overlap)):
        for variant, dets in detect_fn(image.crop(tile)).items():
            for d in dets:
                box = clip_box(project_box(d.bbox, tile), width, height)
                if box is not None:
                    out[variant].append(replace(d, bbox=box, source=f"{base_source}:tile{i}"))
    return out


def merge_views(detections: list[Detection], width: int, height: int, tile_category_ids,
                overlap: float = TILE_OVERLAP, edge_margin: float = TILE_EDGE_MARGIN, drop_truncated: bool = True,
                tile_min_score: float = 0.0, tile_max_area_ratio: float | None = None,
                nms_iou: float = NMS_IOU) -> list[Detection]:
    """Global detections + tile detections of the selected (small/medium) categories, minus boxes cut by an inner
    tile border, then per-class NMS across views.

    Tile views have their own score scale, so they get their own threshold (`tile_min_score`); `tile_max_area_ratio`
    keeps only the small/medium boxes that tiling exists to rescue. Global detections pass through untouched."""
    tiles = tile_grid(width, height, overlap)
    kept = []
    for d in detections:
        idx = view_index(d.source)
        if idx is not None:
            if d.category_id not in tile_category_ids or d.score < tile_min_score:
                continue
            x1, y1, x2, y2 = d.bbox
            if tile_max_area_ratio is not None and (x2 - x1) * (y2 - y1) / (width * height) >= tile_max_area_ratio:
                continue
            if drop_truncated and touches_inner_edge(d.bbox, tiles[idx], width, height, edge_margin):
                continue
        kept.append(d)
    return per_class_nms(kept, nms_iou)


# --------------------------------------------------------------------------- detector


class OwlV2Detector:
    """OWLv2 text-prompted detector. detect(): EXP-01 baseline over the official names; detect_variants(): the four
    EXP-02A variants from one forward pass over the official names plus LVIS_PROMPT_ALIASES."""

    def __init__(self, categories: dict[int, str], model_id: str = OWLV2_BASE, device: str | None = None,
                 min_score: float = DEFAULT_MIN_SCORE, nms_iou: float = NMS_IOU, aliases: dict[int, dict] | None = None):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        device = device or resolve_device()
        self.name = f"owlv2:{model_id}"
        self.model_id, self.device, self.min_score, self.nms_iou = model_id, device, min_score, nms_iou
        self.category_ids = sorted(categories)
        self.prompts = [PROMPT_TEMPLATE.format(categories[c]) for c in self.category_ids]
        self.query_table = build_query_table(aliases if aliases is not None else build_prompt_aliases(categories))
        self.blocks = build_blocks(categories)
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

    def forward_scores(self, image) -> tuple[np.ndarray, np.ndarray]:
        """One forward pass over the whole alias query table: sigmoid scores [boxes, queries] and xyxy boxes in
        image pixels (same box decoding as HF post_process_object_detection on the padded square)."""
        image = image.convert("RGB")
        side = max(image.size)
        inputs = self.processor(text=[list(self.query_table.prompts)], images=image,
                                return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        scores = self._torch.sigmoid(outputs.logits[0]).float().cpu().numpy()
        cx, cy, w, h = outputs.pred_boxes[0].float().cpu().numpy().T
        boxes = np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=1) * np.float32(side)
        return scores, boxes

    def detect_variants(self, image) -> dict[str, list[Detection]]:
        """The four EXP-02A variants (single/chunked x raw/alias) from a single forward pass."""
        width, height = image.size
        scores, boxes = self.forward_scores(image)
        return extract_variants(scores, boxes, self.query_table, self.category_ids, self.blocks, self.min_score,
                                width, height, self.nms_iou)
