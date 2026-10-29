**Full comparison: CAL (in-sample) and VAL (held out, never used for any choice).** 95% CIs come from an image-level bootstrap (1,000 resamples, SEED 42).

| arm | CAL Box F1 | VAL Box F1 | VAL P | VAL R | VAL R_small |
|---|---|---|---|---|---|
| EXP-01 baseline | 0.6576 | 0.6389 | 0.679 | 0.603 | 0.461 |
| EXP-02A chunked prompting | 0.6596 | 0.6364 | 0.674 | 0.603 | 0.461 |
| EXP-02B + selective tiling | 0.6669 | 0.6416 | 0.641 | 0.642 | 0.519 |
| **EXP-02C calibrated ensemble + WBF** | **0.7273** | **0.6822** | **0.685** | **0.680** | **0.536** |

**Paired deltas on VAL.**
- **EXP-02C − EXP-01: ΔF1 +0.0434 [+0.013, +0.080]**, significant. Recall +7.7 pp and R_small +7.5 pp are significant; precision is unchanged (+0.5 pp, n.s.).
- **EXP-02C − EXP-02B: ΔF1 +0.0406 [+0.009, +0.079]**, significant. Precision +4.4 pp and recall +3.8 pp are both significant.
  - This is the result T3 could not get: the ensemble turns tiling's recall into an F1 gain *without* paying for it in precision.

**Overfitting check.**
- EXP-02C has the largest CAL → VAL drop: 0.045, against 0.019–0.025 for the other arms. It fits the most parameters on CAL: 4 calibration anchors, the member-set choice, the fused-score threshold and 2 group thresholds.
- About half of its CAL advantage is in-sample optimism. The held-out gain is still significant against both references, so the improvement is real, but **+0.04 is the honest effect size, not +0.07**.

**Master-plan hypothesis (≥ +8 Box F1 points over EXP-01):** not met. The gain is +7.0 on CAL (in-sample) and **+4.3 on VAL**.
