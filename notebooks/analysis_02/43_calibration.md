**Per-family, per-view calibration.**

*Method.*
- For every (detector, view) pair, a monotone piecewise-linear quantile map is fitted on CAL scores.
- The anchors are: score floor → 0, CAL F1-optimal threshold τ → **0.5 exactly**, CAL maximum → 1. Twenty empirical quantile knots sit on each side of τ.
- Tile maps are fitted only on the tile detections that survive the same filter the fusion applies: tiling subset, size gate and inner-edge truncation.

*What the data say about the "different score scales" premise.*
- The F1-optimal thresholds are **not** far apart. τ = 0.175 for OWLv2 Base global, OWLv2 Large global and both MM-GDINO views; τ = 0.25 / 0.225 for the OWLv2 tile views.
- What differs is the **shape** of the distributions. 48% of MM-GDINO's global detections sit above τ, against 19% for OWLv2. The medians are 0.170 vs 0.087.
- A raw score of 0.3 therefore means "top 22%" for MM-GDINO but "top 7%" for OWLv2 (measured on CAL global detections). The quantile map aligns these ranks, which a single global threshold cannot do.

*The plot.*
- Every curve crosses 0.5 exactly at its own τ (dots).
- The OWLv2 tile curves are shifted right: a tile box needs a higher raw score to count as confident, which is the per-view effect found in T3.
