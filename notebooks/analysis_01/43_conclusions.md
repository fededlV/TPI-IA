---
## Conclusions — EXP-01

**Hypothesis verdict: supported.** A contract-verified baseline with exact federated evaluation and a deterministic QA engine yields a valid, leakage-free submission and a solid reference point:

| Criterion (S2) | Result |
|---|---|
| 4 EDA analyses with actionable conclusions | Done (scale, federated policy, density/overlap, QA benchmark) |
| Baseline Box F1 ≥ 0.45 on development | **0.652** on DEV-120 (CAL 0.658 / VAL 0.639), Mask F1 0.634 |
| Deterministic QA | 40/40 on GT scenes; 30/40 on predicted scenes; LLM-mode agreement 100% |
| Submission #1, 80 rows, verification suite | `submissions/submission_exp01_baseline.csv`, 7 PASS / 1 SKIPPED (VRAM, CPU run) / **0 FAIL** |

**What we learned (and what it decides):**
1. **Recall on small objects is the bottleneck** (0.50 vs 0.74–0.79 on medium/large; ~69% of all misses). → EXP-02B: 2×2 tiled inference with 25% overlap.
2. **Some misses are grounding failures, not scale failures** (`elephant`, `strap`, `cabinet`). → EXP-02A: LVIS aliases + chunked prompts.
3. **Verified-negative FPs are rare** (5 in 120 images) and the federated policy rewards a permissive threshold. → EXP-02C: calibrated multi-family ensemble tuned for recall, with anti-collision WBF (same-source fusion only at IoU ≥ 0.80, justified by the 0.775 max intra-class GT overlap).
4. **SAM2 is not yet the limiting factor** (Box→Mask F1 gap 0.018, 0 fallbacks). → EXP-03 targets the residual composite-object cases.
5. **QA errors are 100% perception-propagated**; counting is the most fragile answer type (4/10). → improvements in detection recall/duplicate control should translate directly into QA gains.

**Limitations (stated explicitly):** CPU-only run, so latencies are not representative of the Kaggle budget and VRAM is not measured; per-split QA accuracies rest on 20 questions each; the threshold is global (one parameter), deliberately leaving per-class calibration to EXP-02C.
