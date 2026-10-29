"""SceneJSON construction: column-major RLE, object geometry, dominant color, relations and schema validation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

from src.config_and_seeds import size_from_area_ratio

REQUIRED_OBJECT_KEYS = ("id", "category_id", "confidence", "bbox", "mask", "centroid", "area_ratio",
                        "position", "size")
MAX_OBJECTS = 300
MAX_RELATION_OBJECTS = 95
MAX_RELATIONS = 20000  # schema maxItems; we stay strictly below it

POSITION_GRID = (
    ("top_left", "top_center", "top_right"),
    ("middle_left", "center", "middle_right"),
    ("bottom_left", "bottom_center", "bottom_right"),
)


# --------------------------------------------------------------------------- RLE


def encode_rle(mask: np.ndarray) -> dict:
    """Uncompressed column-major RLE; counts always start with the run of zeros (possibly 0)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-D, got shape {mask.shape}")
    h, w = mask.shape
    flat = mask.ravel(order="F").astype(np.int8)
    change_points = np.flatnonzero(np.diff(flat)) + 1
    boundaries = np.concatenate(([0], change_points, [flat.size]))
    counts = np.diff(boundaries).tolist()
    if flat[0] == 1:
        counts.insert(0, 0)
    return {"size": [int(h), int(w)], "counts": [int(c) for c in counts]}


def decode_rle(rle: dict) -> np.ndarray:
    h, w = (int(v) for v in rle["size"])
    counts = np.asarray(rle["counts"], dtype=np.int64)
    if counts.size == 0 or (counts < 0).any() or int(counts.sum()) != h * w:
        raise ValueError(f"RLE counts must be non-negative and sum to H*W={h * w}")
    values = np.arange(counts.size) % 2 == 1
    flat = np.repeat(values, counts)
    return flat.reshape((h, w), order="F")


# --------------------------------------------------------------------------- masks & boxes


def xywh_to_xyxy(bbox: Iterable[float]) -> list[float]:
    x, y, w, h = (float(v) for v in bbox)
    return [x, y, x + w, y + h]


def polygon_to_mask(segmentation: list[list[float]], height: int, width: int) -> np.ndarray:
    """Rasterize COCO polygon segmentation (list of flat [x1, y1, x2, y2, ...] rings)."""
    canvas = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for ring in segmentation:
        if len(ring) >= 6:
            draw.polygon([(ring[i], ring[i + 1]) for i in range(0, len(ring) - 1, 2)], fill=1, outline=1)
    return np.array(canvas, dtype=bool)


def clip_bbox(bbox_xyxy: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    return [min(max(x1, 0.0), width), min(max(y1, 0.0), height),
            min(max(x2, 0.0), width), min(max(y2, 0.0), height)]


def ensure_non_empty_mask(mask: np.ndarray, bbox_xyxy: Iterable[float]) -> tuple[np.ndarray, bool]:
    """Guarantee sum(mask) >= 1: fall back to the clipped bbox fill, then to the nearest pixel."""
    mask = np.asarray(mask, dtype=bool)
    if mask.any():
        return mask, False
    h, w = mask.shape
    x1, y1, x2, y2 = clip_bbox(bbox_xyxy, w, h)
    filled = np.zeros_like(mask)
    filled[int(np.floor(y1)):int(np.ceil(y2)), int(np.floor(x1)):int(np.ceil(x2))] = True
    if not filled.any():
        cx = int(np.clip((x1 + x2) / 2, 0, w - 1))
        cy = int(np.clip((y1 + y2) / 2, 0, h - 1))
        filled[cy, cx] = True
    return filled, True


def mask_tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Half-open pixel bbox (x1, y1, x2, y2) of a non-empty mask."""
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


# --------------------------------------------------------------------------- geometry & color


def position_from_centroid(cx: float, cy: float, width: int, height: int) -> str:
    col = min(max(int(3 * cx / width), 0), 2)
    row = min(max(int(3 * cy / height), 0), 2)
    return POSITION_GRID[row][col]


def compute_object_geometry(mask: np.ndarray) -> dict:
    """Centroid in absolute pixel indices [cx, cy], area_ratio, 3x3 position and size."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("Geometry requires a non-empty mask")
    cx, cy = float(xs.mean()), float(ys.mean())
    area_ratio = float(xs.size) / float(h * w)
    return {
        "centroid": [cx, cy],
        "area_ratio": area_ratio,
        "position": position_from_centroid(cx, cy, w, h),
        "size": size_from_area_ratio(area_ratio),
    }


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


_HUE_BINS = ((15, "red"), (40, "orange"), (70, "yellow"), (160, "green"), (200, "cyan"),
             (255, "blue"), (290, "purple"), (335, "pink"), (360, "red"))


def _name_hsv_pixels(hsv: np.ndarray) -> np.ndarray:
    """Name each pixel from PIL HSV (all channels 0..255)."""
    h = hsv[:, 0].astype(np.float64) * 360.0 / 255.0
    s = hsv[:, 1] / 255.0
    v = hsv[:, 2] / 255.0
    names = np.empty(len(hsv), dtype=object)
    edges = np.array([b[0] for b in _HUE_BINS])
    labels = np.array([b[1] for b in _HUE_BINS], dtype=object)
    names[:] = labels[np.minimum(np.searchsorted(edges, h, side="right"), len(labels) - 1)]
    achromatic = s < 0.20
    names[achromatic] = np.where(v[achromatic] > 0.80, "white", "gray")
    names[v < 0.20] = "black"
    return names


def dominant_color(image_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Dominant named color (HSV vote) and mean CIE LAB over a 1px-eroded copy of the mask."""
    eroded = binary_erosion(mask, iterations=1)
    region = eroded if eroded.any() else mask
    rgb = np.asarray(image_rgb, dtype=np.uint8)[region]
    if rgb.size == 0:
        return {"name": "unknown", "lab": [0.0, 0.0, 0.0], "eroded": False}
    hsv = np.array(Image.fromarray(rgb.reshape(-1, 1, 3)).convert("HSV")).reshape(-1, 3)
    names, votes = np.unique(_name_hsv_pixels(hsv), return_counts=True)
    lab = _srgb_to_lab(rgb).mean(axis=0)
    return {"name": str(names[int(np.argmax(votes))]), "lab": [round(float(v), 2) for v in lab],
            "eroded": bool(eroded.any())}


# --------------------------------------------------------------------------- objects & scene


def build_object(object_id: str, category_id: int, confidence: float, bbox_xyxy: Iterable[float],
                 mask: np.ndarray, image_rgb: np.ndarray | None = None,
                 category_name: str | None = None, **extra) -> dict:
    """Build one SceneJSON object with the 9 required keys (plus optional attributes)."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    bbox = clip_bbox(bbox_xyxy, w, h)
    mask, used_fallback = ensure_non_empty_mask(mask, bbox)
    obj = {
        "id": str(object_id),
        "category_id": int(category_id),
        "confidence": float(min(max(confidence, 0.0), 1.0)),
        "bbox": bbox,
        "mask": encode_rle(mask),
        **compute_object_geometry(mask),
    }
    if category_name is not None:
        obj["category_name"] = category_name
    if image_rgb is not None:
        color = dominant_color(image_rgb, mask)
        obj["dominant_color"] = color["name"]
        obj["dominant_color_lab"] = color["lab"]
    if used_fallback:
        obj["mask_fallback"] = "bbox"
    obj.update(extra)
    return obj


def _boxes_intersect(a: Iterable[float], b: Iterable[float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)


def _masks_intersect(ma: np.ndarray, mb: np.ndarray, ta: tuple, tb: tuple) -> bool:
    x1, y1 = max(ta[0], tb[0]), max(ta[1], tb[1])
    x2, y2 = min(ta[2], tb[2]), min(ta[3], tb[3])
    if x2 <= x1 or y2 <= y1:
        return False
    return bool((ma[y1:y2, x1:x2] & mb[y1:y2, x1:x2]).any())


def build_relations(objects: list[dict], max_objects: int = MAX_RELATION_OBJECTS,
                    max_relations: int = MAX_RELATIONS) -> list[dict]:
    """Full symmetric relation graph by pure centroid comparison plus bbox/mask overlaps.

    Only the top-`max_objects` by confidence participate; symmetric pairs are emitted together and
    the list is capped strictly below `max_relations` (pairs of higher-confidence objects first).
    """
    ranked = sorted(objects, key=lambda o: -o["confidence"])[:max_objects]
    masks = [decode_rle(o["mask"]) for o in ranked]
    tight = [mask_tight_bbox(m) for m in masks]
    relations: list[dict] = []

    def add_pair(s: str, p: str, o: str, inverse: str) -> bool:
        if len(relations) + 2 >= max_relations:
            return False
        relations.append({"subject": s, "predicate": p, "object": o})
        relations.append({"subject": o, "predicate": inverse, "object": s})
        return True

    for i in range(len(ranked)):
        a = ranked[i]
        for j in range(i + 1, len(ranked)):
            b = ranked[j]
            (ax, ay), (bx, by) = a["centroid"], b["centroid"]
            pairs = []
            if ax != bx:
                pairs.append(("left_of", "right_of") if ax < bx else ("right_of", "left_of"))
            if ay != by:
                pairs.append(("above", "below") if ay < by else ("below", "above"))
            if _boxes_intersect(a["bbox"], b["bbox"]) or _masks_intersect(masks[i], masks[j], tight[i], tight[j]):
                pairs.append(("overlaps", "overlaps"))
            for pred, inv in pairs:
                if not add_pair(a["id"], pred, b["id"], inv):
                    return relations
    return relations


def build_scene_json(image_id: str, width: int, height: int, objects: list[dict],
                     questions: list[dict] | None = None, questions_llm: list[dict] | None = None) -> dict:
    """Assemble a SceneJSON row; objects are ranked by confidence and capped at 300."""
    kept = sorted(objects, key=lambda o: -o["confidence"])[:MAX_OBJECTS]
    scene = {
        "image": {"id": str(image_id), "width": int(width), "height": int(height)},
        "objects": kept,
        "relations": build_relations(kept),
    }
    if questions is not None:
        scene["questions"] = questions
    if questions_llm is not None:
        scene["questions_llm"] = questions_llm
    return scene


def validate_scene(scene: dict, schema: dict) -> list[str]:
    """Return human-readable jsonschema errors (empty list means valid)."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in validator.iter_errors(scene)]
