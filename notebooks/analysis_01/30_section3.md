---
## Section 3 — Traceable execution of the EXP-01 baseline (120 dev + 60 test images)

Inference was executed with `scripts/run_exp01_inference.py`, which calls exactly the same `src/` functions used below, records an `ExecutionTracer` stage per image (`duration_ms`, input/output counts, warnings, object lineage), and caches results under `artifacts/cache_exp01/`:

1. `detect` — OWLv2 Base over all 180 images, raw detections kept at a low floor (score ≥ 0.05) so the operating threshold can be tuned without re-running the model.
2. `segment --threshold T` — SAM2.1 Base+ on the boxes surviving the CAL-tuned threshold `T`, then `SceneJSON` construction.

This notebook loads those caches (≈65 CPU-minutes of inference), **re-runs fresh inference on sample images to prove the cache reproduces the model**, and then evaluates.
