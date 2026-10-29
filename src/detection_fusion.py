"""EXP-02C: per-detector / per-view monotonic score calibration and the calibrated multi-family fusion.

Every detector view gets its own quantile map fitted on CAL that sends its CAL F1-optimal threshold to the canonical
score 0.5, so scores of OWLv2 (low sigmoids) and Grounding-DINO (higher) become comparable before fusion.
Detection sources read "<detector>:<view>" ("owlv2:global", "mmgdino:tile2").
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.perception_detectors import Detection, view_index

CANONICAL_SCORE = 0.5
CALIBRATION_KNOTS = 20  # quantile knots on each side of tau


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
