**Analysis — explicit sibling suppression.**

- **The explicit rule is a no-op at every operating point.**
  - `suppress_siblings` (IoU ≥ 0.7) removes **0 detections** at the CAL-optimal threshold in all 8 arms, so F1 is unchanged.
  - It only fires below the operating point (9–18 suppressions at the 0.05 floor).
- **Why.** Siblings share a block, so the per-block argmax already prevents two siblings from labelling the same predicted box. Two *different* boxes with sibling labels overlapping at IoU ≥ 0.7 only occur among low-scoring candidates. The disambiguation work is done by the block design, not by post-hoc NMS.
- **The real residual error is a wrong winner, not a duplicate.**
  - Of the 155 sibling-class GT instances in CAL, 7 (Base) / 10 (Large) are missed while a sibling label correctly localises them (IoU ≥ 0.5): a `boot` called `shoe`, a `jean` called `trousers`.
  - Suppression cannot recover those. It would need re-ranking with a different signal (e.g. a size prior for `watch` vs. `clock`, or context). At 4.5% of sibling instances it is a minor error source compared with small-object misses.
- **Decision.** The rule stays in the code (it is free and guards T4's multi-detector fusion, where siblings from *different* detectors can overlap), but it is not a lever for EXP-02.
