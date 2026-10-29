"""EXP-02C: per-detector / per-view monotonic score calibration and the calibrated multi-family fusion.

Every detector view gets its own quantile map fitted on CAL that sends its CAL F1-optimal threshold to the canonical
score 0.5, so scores of OWLv2 (low sigmoids) and Grounding-DINO (higher) become comparable before fusion.
Detection sources read "<detector>:<view>" ("owlv2:global", "mmgdino:tile2").

Fusion is Weighted Boxes Fusion with an intra-detector non-self-collision rule: two confident boxes of the same
detector and view never merge unless they overlap almost completely, so adjacent instances (chairs, plates, shoes)
survive. Post-fusion thresholds are set per LVIS sensitivity group (standard / micro / fp_prone), all fitted on CAL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from src.lvis_evaluator import box_iou
from src.perception_detectors import (MAX_DETECTIONS, NMS_IOU, TILE_EDGE_MARGIN, Detection, GroundingDinoDetector,
                                      OwlV2Detector, detect_with_tiles, filter_tile_views, per_class_nms, view_index)

CANONICAL_SCORE = 0.5
CALIBRATION_KNOTS = 20  # quantile knots on each side of tau
WBF_IOU = 0.55  # same-category proposals at or above this IoU (vs the fused box) may fuse
SELF_COLLISION_IOU = 0.80  # two confident boxes of one detector view only fuse at or above this IoU
GROUPS = ("standard", "micro", "fp_prone")


def detector_of(source: str) -> str:
    return source.split(":", 1)[0]


def calibration_key(source: str) -> str:
    """'<detector>:global' or '<detector>:tile' (the four tiles share one map)."""
    return f"{detector_of(source)}:{'global' if view_index(source) is None else 'tile'}"


def retag(detections: list[Detection], detector: str) -> list[Detection]:
    """Rename the detector part of the sources (two OWLv2 backbones in one ensemble need distinct names)."""
    return [replace(d, source=f"{detector}:{d.source.split(':', 1)[1]}") for d in detections]


@dataclass(frozen=True)
class QuantileCalibrator:
    """Piecewise-linear monotone map through (floor, 0), quantile knots, (tau, 0.5), quantile knots, (max, 1)."""
    knots_x: tuple[float, ...]
    knots_y: tuple[float, ...]

    @classmethod
    def fit(cls, scores, tau: float, floor: float, n_knots: int = CALIBRATION_KNOTS) -> QuantileCalibrator:
        s = np.asarray(scores, dtype=float)
        s = s[s >= floor]
        if s.size == 0:
            raise ValueError("no scores above the floor to fit a calibrator")
        if not floor < tau < s.max():
            raise ValueError(f"tau={tau} must lie strictly between the floor {floor} and the max score {s.max()}")
        q = np.linspace(0.0, 1.0, n_knots + 1)[1:-1]
        below, above = s[s < tau], s[s >= tau]
        knots = [(floor, 0.0, True)]
        knots += [(x, CANONICAL_SCORE * p, False) for x, p in zip(np.quantile(below, q), q)] if below.size else []
        knots += [(tau, CANONICAL_SCORE, True)]
        knots += [(x, CANONICAL_SCORE + (1 - CANONICAL_SCORE) * p, False) for x, p in zip(np.quantile(above, q), q)]
        knots += [(float(s.max()), 1.0, True)]
        kept = [knots[0]]
        for x, y, anchor in knots[1:]:  # ties: keep x strictly increasing; an anchor replaces a tied quantile knot
            if x > kept[-1][0]:
                kept.append((x, y, anchor))
            elif anchor and not kept[-1][2]:
                kept[-1] = (x, y, anchor)
        return cls(tuple(float(k[0]) for k in kept), tuple(float(k[1]) for k in kept))

    def __call__(self, score):
        out = np.interp(score, self.knots_x, self.knots_y, left=0.0, right=1.0)
        return float(out) if np.ndim(out) == 0 else out

    def to_dict(self) -> dict:
        return {"knots_x": list(self.knots_x), "knots_y": list(self.knots_y)}

    @classmethod
    def from_dict(cls, d: dict) -> QuantileCalibrator:
        return cls(tuple(float(v) for v in d["knots_x"]), tuple(float(v) for v in d["knots_y"]))


def calibrate(detections: list[Detection], calibrators: dict[str, QuantileCalibrator]) -> list[Detection]:
    """Canonical scores per detector view; the raw score is kept for lineage. Unknown views fail fast."""
    return [replace(d, score=calibrators[calibration_key(d.source)](d.score), raw_score=d.score) for d in detections]


# --------------------------------------------------------------------------- anti-collision WBF


def _ranked(detections):
    return sorted(detections, key=lambda d: (-d.score, d.category_id, d.bbox, d.source))


@dataclass
class _Cluster:
    members: list[Detection]
    box: tuple[float, float, float, float]

    def add(self, det: Detection) -> None:
        self.members.append(det)
        w = np.array([max(m.score, 1e-6) for m in self.members])
        boxes = np.array([m.bbox for m in self.members], dtype=float)
        self.box = tuple(float(v) for v in (w[:, None] * boxes).sum(axis=0) / w.sum())

    def collides(self, det: Detection, self_iou_thr: float, high_conf: float) -> bool:
        return det.score >= high_conf and any(
            m.source == det.source and m.score >= high_conf and box_iou(list(m.bbox), list(det.bbox)) < self_iou_thr
            for m in self.members)


def weighted_boxes_fusion(detections: list[Detection], n_detectors: int, iou_thr: float = WBF_IOU,
                          self_iou_thr: float = SELF_COLLISION_IOU, high_conf: float = CANONICAL_SCORE,
                          anti_collision: bool = True) -> list[Detection]:
    """Per category, in descending calibrated score: each box joins the cluster whose fused box it overlaps most
    (IoU >= iou_thr) unless that cluster already holds a confident box of the same detector view that it does not
    nearly duplicate (the non-self-collision rule); otherwise it opens a new cluster.

    Fused box = score-weighted mean of the members. Fused score = sum over detectors of their best member's score /
    n_detectors: views of one detector count once, and a box found by one family alone is down-weighted."""
    detectors = {detector_of(d.source) for d in detections}
    if len(detectors) > n_detectors:
        raise ValueError(f"{len(detectors)} detectors in the input ({sorted(detectors)}) > n_detectors={n_detectors}")
    clusters: dict[int, list[_Cluster]] = {}
    for det in _ranked(detections):
        best, best_iou = None, iou_thr
        for c in clusters.setdefault(det.category_id, []):
            iou = box_iou(list(c.box), list(det.bbox))
            if iou >= best_iou and not (anti_collision and c.collides(det, self_iou_thr, high_conf)):
                if best is None or iou > best_iou:
                    best, best_iou = c, iou
        if best is None:
            clusters[det.category_id].append(_Cluster([det], det.bbox))
        else:
            best.add(det)
    fused = []
    for cid, cs in clusters.items():
        for c in cs:
            per_detector: dict[str, Detection] = {}
            for m in c.members:  # members arrive in descending score: the first one per detector is its best
                per_detector.setdefault(detector_of(m.source), m)
            top = c.members[0]
            fused.append(Detection(cid, min(1.0, sum(m.score for m in per_detector.values()) / n_detectors), c.box,
                                   "wbf:" + "+".join(sorted(per_detector)), top.raw_score))
    return sorted(fused, key=lambda d: (-d.score, d.category_id, d.bbox))


# --------------------------------------------------------------------------- LVIS sensitivity groups


def select_fp_prone(neg_fp: dict[int, int], evaluated: dict[int, int], min_count: int = 3,
                    min_share: float = 0.5) -> set[int]:
    """Categories whose CAL predictions are mostly verified false positives (category in the image's neg list)."""
    return {cid for cid, n in neg_fp.items() if n >= min_count and n / max(evaluated.get(cid, 0), 1) >= min_share}


def assign_sensitivity_groups(category_ids, micro_ids, fp_prone_ids) -> dict[int, str]:
    """fp_prone > micro > standard."""
    return {cid: "fp_prone" if cid in fp_prone_ids else "micro" if cid in micro_ids else "standard"
            for cid in category_ids}


def apply_group_thresholds(detections: list[Detection], group_of: dict[int, str],
                           thresholds: dict[str, float]) -> list[Detection]:
    return [d for d in detections if d.score >= thresholds[group_of.get(d.category_id, "standard")]]


# --------------------------------------------------------------------------- frozen EXP-02C configuration


@dataclass(frozen=True)
class MemberSpec:
    """One ensemble detector: family ('owlv2' | 'gdino'), checkpoint and the prompt variant it is read with."""
    family: str
    model_id: str
    variant: str


@dataclass(frozen=True)
class EnsembleConfig:
    members: dict[str, MemberSpec]
    calibrators: dict[str, QuantileCalibrator]
    tile_category_ids: frozenset[int]
    tile_max_area_ratio: float | None
    group_of: dict[int, str]
    thresholds: dict[str, float]
    drop_truncated: bool = True
    edge_margin: float = TILE_EDGE_MARGIN
    min_calibrated: float = 0.0  # calibrated floor before fusion
    fusion: str = "wbf"  # "wbf" | "nms" (calibrated union + per-class NMS, ablation)
    iou_thr: float = WBF_IOU
    self_iou_thr: float = SELF_COLLISION_IOU
    high_conf: float = CANONICAL_SCORE
    anti_collision: bool = True
    max_detections: int = MAX_DETECTIONS
    notes: dict = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict:
        return {"members": {k: vars(m) for k, m in self.members.items()},
                "calibrators": {k: c.to_dict() for k, c in self.calibrators.items()},
                "tile_category_ids": sorted(self.tile_category_ids), "tile_max_area_ratio": self.tile_max_area_ratio,
                "group_of": {str(k): v for k, v in sorted(self.group_of.items())}, "thresholds": self.thresholds,
                "drop_truncated": self.drop_truncated, "edge_margin": self.edge_margin,
                "min_calibrated": self.min_calibrated, "fusion": self.fusion, "iou_thr": self.iou_thr,
                "self_iou_thr": self.self_iou_thr, "high_conf": self.high_conf, "anti_collision": self.anti_collision,
                "max_detections": self.max_detections, "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict) -> EnsembleConfig:
        d = dict(d)
        d["members"] = {k: MemberSpec(**m) for k, m in d["members"].items()}
        d["calibrators"] = {k: QuantileCalibrator.from_dict(c) for k, c in d["calibrators"].items()}
        d["tile_category_ids"] = frozenset(int(c) for c in d["tile_category_ids"])
        d["group_of"] = {int(k): v for k, v in d["group_of"].items()}
        unknown = set(d["thresholds"]) - set(GROUPS)
        if unknown:
            raise ValueError(f"unknown sensitivity groups: {unknown}")
        return cls(**d)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> EnsembleConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fuse_views(raw_by_detector: dict[str, list[Detection]], width: int, height: int,
               config: EnsembleConfig) -> list[Detection]:
    """Blind EXP-02C post-processing before thresholds: per detector, tile filter (subset, size gate, inner-edge
    truncation) and calibration; then anti-collision WBF (or NMS) across detectors and views."""
    unknown = set(raw_by_detector) - set(config.members)
    if unknown:
        raise ValueError(f"detections from detectors outside the ensemble: {sorted(unknown)}")
    pool = []
    for name, dets in raw_by_detector.items():
        if any(detector_of(d.source) != name for d in dets):
            raise ValueError(f"detections under {name!r} carry another detector's source; use retag()")
        kept = filter_tile_views(dets, width, height, config.tile_category_ids, edge_margin=config.edge_margin,
                                 drop_truncated=config.drop_truncated, tile_max_area_ratio=config.tile_max_area_ratio)
        pool += [d for d in calibrate(kept, config.calibrators) if d.score >= config.min_calibrated]
    if config.fusion == "nms":
        return per_class_nms(pool, NMS_IOU)
    if config.fusion != "wbf":
        raise ValueError(f"unknown fusion mode {config.fusion!r}")
    return weighted_boxes_fusion(pool, len(config.members), config.iou_thr, config.self_iou_thr, config.high_conf,
                                 config.anti_collision)


def fuse_image(raw_by_detector: dict[str, list[Detection]], width: int, height: int,
               config: EnsembleConfig) -> list[Detection]:
    """fuse_views + per-group thresholds + the top-`max_detections` cap: the final EXP-02C detections."""
    fused = apply_group_thresholds(fuse_views(raw_by_detector, width, height, config), config.group_of,
                                   config.thresholds)
    return fused[:config.max_detections]


# --------------------------------------------------------------------------- live (blind) ensemble


class EnsembleDetector:
    """Runs every member through the 2x2 tiled pass (views tagged '<member>:<view>') and fuses them with the frozen
    EXP-02C config. `families` maps member name -> object with detect_variants(image) (OwlV2Detector,
    GroundingDinoDetector)."""

    def __init__(self, families: dict, config: EnsembleConfig):
        if set(families) != set(config.members):
            raise ValueError(f"members {sorted(families)} do not match the config {sorted(config.members)}")
        self.families, self.config = families, config
        self.name = "ensemble:" + "+".join(sorted(families))

    @classmethod
    def from_config(cls, config: EnsembleConfig, categories: dict[int, str], device: str | None = None):
        families = {}
        for name, spec in config.members.items():
            if spec.family == "owlv2":
                families[name] = OwlV2Detector(categories, model_id=spec.model_id, device=device)
            else:
                families[name] = GroundingDinoDetector(categories, model_id=spec.model_id, device=device,
                                                       use_aliases=spec.variant == "chunked_alias")
        return cls(families, config)

    def detect(self, image) -> list[Detection]:
        image = image.convert("RGB")
        raw = {}
        for name, family in self.families.items():
            variant = self.config.members[name].variant
            raw[name] = detect_with_tiles(image, family.detect_variants, base_source=name)[variant]
        return fuse_image(raw, *image.size, self.config)
