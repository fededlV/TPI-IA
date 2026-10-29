# EXP-02A/B — Chunked Prompting, LVIS Aliases and Selective 2×2 Tiling (Sections 1–2)

**TPI Visual Scene Understanding — UTN Artificial Intelligence, Spring 2026 — Group 2**

**Problem.** EXP-01 (OWLv2 Base, the 120 raw names in one prompt) reached CAL Box F1 = 0.6576 but only **0.516 box recall on small objects**, and 71.06% of the 1,790 GT instances have `area_ratio < 0.01` (median 0.00289: rings, watches, doorknobs, handles, wall sockets, forks, knives, cell phones…). Perception is 40% of the `PipelineScore`, and small objects make up most of its false negatives.

**Hypotheses.**
- **EXP-02A:** splitting the vocabulary into 5 semantic blocks of 24 classes (*Chunked Prompting*) and adding visual aliases for archaic/ambiguous LVIS names (`LVIS_PROMPT_ALIASES`, remapped to the official `category_id`) raises recall without hurting precision.
- **EXP-02B:** an additional pass over 2×2 overlapping tiles (25% overlap), restricted to small/medium categories, with boxes projected back to `[0, W] × [0, H]` and truncation artifacts on inner tile borders removed, raises recall on small objects (`area_ratio < 0.0035`).

**Controlled design.**
- Both ablations run on `CAL` (80 images, 1,231 GT instances). Each one changes one variable at a time against its own control arm.
- Every arm gets its own CAL-optimal score threshold, so arms are compared at their best operating point, not at a threshold inherited from another arm.
- `VAL` is never consulted here; the held-out comparison belongs to T4 (Section 3).

**Evaluation.**
- Box-only LVIS federated matching (IoU ≥ 0.5), the exact rule of `lvis_evaluator.evaluate_federated`, computed per image (`box_match_details`).
- *Statistically significant* means the 95% CI of a paired image-level bootstrap (1,000 resamples, `SEED = 42`) excludes 0.

**Compute.**
- The detections were produced on a **Kaggle Tesla T4** by `scripts/run_exp02_inference.py` (OWLv2 Base and OWLv2 Large) and cached in `artifacts/cache_exp02/`.
- This notebook only reads those caches, so it runs on CPU in under a minute. `annotations.json` is used **only** for evaluation; the runner never opens it.
