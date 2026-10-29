## Conclusions of EXP-02A/B and hand-off to T4

| question | answer (CAL, 80 images, 95% paired bootstrap) |
|---|---|
| Does chunked prompting (5 × 24) help OWLv2? | Recall +0.65 pp (significant), F1 +0.002 (n.s.). Free at inference. **Kept.** |
| Do LVIS visual aliases help OWLv2? | Precision −1.2 to −1.8 pp (significant), recall n.s. **Dropped by default.** |
| Is the explicit sibling rule needed? | 0 suppressions at the operating point; the per-block argmax already disambiguates. Kept as a free guard. |
| Does selective 2×2 tiling help small objects? | **Small recall +6.4 pp [+4.6, +8.2]**, medium +1.4 pp, large unchanged; F1 +0.007 (n.s.). **Kept.** |
| S3 vs. EXP-01 | **Small recall +6.9 pp [+5.1, +8.7] ✔**; Box F1 +0.0093 [−0.0024, +0.0210] ✘ (positive, not significant at 95%). **Partially met.** |

**Final EXP-02A/B configuration.**
- Detector and prompting: OWLv2 Base, `chunked_raw`, global threshold 0.175.
- Tiling: 2×2 tiles with 25% overlap, tile threshold 0.25, inner-edge truncation filter (3 px), size gate `area_ratio < 0.035`, 106 tiled categories.
- CAL result: P 0.660, R 0.674, F1 0.6669, small recall 0.585.
- Frozen in `artifacts/cache_exp02/exp02ab_summary.json` and `tiling_categories.json`.

**Findings T4 must act on.**
1. **OWLv2 Large is significantly worse than OWLv2 Base on this data** (F1 −0.032, CI [−0.056, −0.008]) and 4.3× slower. The master plan's ensemble assumes Large is the strong OWLv2 member; T4 should treat Base and Large as candidates and decide on CAL/VAL, not by model size.
2. **Precision, not recall, is now the bottleneck.** Tiling's recall gain is cancelled by uncorroborated tile false positives. This is exactly what per-view/per-detector score calibration and WBF fusion (a box confirmed by two views or two detectors is promoted) are designed to solve, and the most likely way to turn the recall gain into a significant F1 gain.
3. **Views have different score scales.** A single threshold across views costs −8.1 pp of large recall (B2). Calibration in T4 must be per detector *and* per view (global vs. tile).
4. **Chunking matters for MM-GDINO for a different reason.** Its BERT encoder truncates at 256 tokens, and the 120 classes do not fit in one prompt. For GDINO the 5 × 24 blocks are a hard requirement, not an optimisation. Aliases may also help there more than with CLIP and deserve a re-test.
5. **Candidate sibling pair** `computer monitor` / `television set`, exposed by the `tv`/`television` aliases.
6. **Caches for T4.** `artifacts/cache_exp02/{base,large}/detections_{variant}.json.gz` cover all 180 images (CAL + VAL + test) with untouched tile views. The VAL comparison and the bootstrap CIs on VAL (T4 Section 3) need no new OWLv2 inference.
