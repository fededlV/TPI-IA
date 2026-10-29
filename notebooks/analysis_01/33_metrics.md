**Analysis — EXP-01 baseline metrics (threshold 0.175, SAM2.1 Base+, `multimask_output=False`).**

| Split | Box P | Box R | **Box F1** | **Mask F1** | LVIS Coverage | objects/img | total ms/img (CPU) |
|---|---|---|---|---|---|---|---|
| CAL (80) | 0.688 | 0.630 | **0.658** | **0.640** | 0.454 | 33.1 | 21,666 |
| VAL (40) | 0.679 | 0.603 | **0.639** | **0.620** | 0.472 | 28.2 | 21,379 |
| DEV (120) | 0.686 | 0.621 | **0.652** | **0.634** | 0.460 | 31.5 | 21,570 |

1. **S2 criterion met:** development Box F1 = **0.652** (≥ 0.45 required; also above the 0.50 Direct-Pass bar). VAL, never used for any decision, is only 0.019 below CAL — the single tuned parameter generalises.
2. **Segmentation costs little at this stage:** Mask F1 is only 0.018 below Box F1, i.e. for ~97% of the boxes that match a GT instance, the single SAM2 mask also matches the GT mask at IoU ≥ 0.5. There were **0 bbox fallbacks** across 5,592 objects: the non-empty guarantee never had to fire, but every row is protected if it does. The residual box→mask loss concentrates on composite/thin objects (handled by EXP-03's box-aware multimask selection).
3. **LVIS coverage 0.46:** 54% of our predictions fall in unverified categories and are ignored by the metric. With ~31.5 predictions/img against 14.9 GT instances/img, the detector "over-predicts", but mostly where the metric does not look; this is why the optimal threshold is low (Section 2, analysis 2).
4. **Precision is not driven by verified negatives:** only 5 predictions over 120 images land in `neg_category_ids` (next cell). The FP budget is therefore spent on unmatched/duplicate boxes in exhaustive positive classes and on localisation errors (IoU < 0.5), not on hallucinated classes.
5. **Cost (CPU):** 11.9 s detection + 9.5 s segmentation + 0.24 s scene construction ≈ **21.6 s/img**; 180 images ≈ 65 CPU-minutes. Scene construction (RLE, geometry, colour, full relation graph) is ~1% of the time, so the representation layer is never the bottleneck. VRAM is **N/A (CPU)**: no GPU was available for this run; the Kaggle T4 run of later experiments will populate this column and suite Test 8.
6. **Test images:** the 60 test scenes receive the same pipeline (30.2 objects/img, similar latency) — no metric is computable without GT, but their distribution of predictions per image matches development, which is a sanity check against a silent failure on the test folder.
