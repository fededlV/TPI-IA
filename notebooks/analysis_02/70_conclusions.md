## Conclusions of EXP-02C (T4) and the S4 verdict

| question | answer (VAL = held out; 95% paired bootstrap) |
|---|---|
| Does a second family add diversity? | Yes. MM-GDINO alone ≈ OWLv2 Base (ΔF1 −0.002, n.s.), but each family finds instances the other misses: oracle union small recall is 0.695 vs 0.585. |
| Does the ensemble beat a single family? | Yes. C6 − C1 = **+0.058 F1** on CAL (significant); precision and recall both improve. |
| Does monotonic calibration help? | Not on its own: under NMS, C4 − C3 is +0.001 (n.s.), because the raw F1-optimal thresholds coincide (τ = 0.175). It is needed so that WBF can *add* scores on a common scale. |
| Does WBF beat NMS? | On recall, yes: +4.8 pp R and +5.8 pp R_small (significant). F1 +0.016, p = 0.051. |
| Does the non-self-collision rule help? | **No measurable effect.** The per-view NMS at 0.5 already prevents same-view collapses (0 colliding pairs out of 64,944). Kept as a tested invariant; no gain is attributed to it. |
| Do sensitivity-group thresholds help? | Slightly: R_small +2.4 pp (significant), F1 +0.004 (n.s.). The `fp_prone` group is empty, since verified-negative FPs are about 1% of FPs. |
| EXP-02C vs EXP-01 (VAL) | **Box F1 +0.043 [+0.013, +0.080]**, Mask F1 +0.046 [+0.018, +0.083], 0.40·Box + 0.35·Mask +0.033 [+0.013, +0.059]. All significant. |
| EXP-02C vs EXP-02B (VAL) | **Box F1 +0.041 [+0.009, +0.079]**, significant. Precision and recall rise together. |

**S4 verdict: MET.**
- VAL Box F1 = **0.6822 > 0.60**, and it is significantly above both EXP-01 and EXP-02B on held-out data.
- `submissions/submission_exp02_perception.csv` is valid: 80 rows, **8/8 PASS** on a T4.

**Deviations and caveats, documented rather than hidden.**
1. **The OWLv2 member is Base, not Large.** It was selected on CAL. Large + MM-GDINO is worse (0.708 vs 0.724), and Large adds nothing as a third member.
2. **The non-self-collision rule** is implemented and tested but is inert in this pipeline (see above).
3. **Optimism.** The gain is +0.070 on CAL but +0.043 on VAL. The VAL figure is the effect size to report.
4. **The master-plan target of "≥ +8 F1 points"** is not reached on held-out data (+4.3).
5. **Cost:** about 66 s/img on a T4, dominated by MM-GDINO's 25 forward passes.

**Hand-off to T5 (EXP-03).**
- Freeze these boxes (`exp02c/config.json`, sha `912f09d78ec5dd6e`; scenes in `exp02c/scenes.jsonl`).
- Change only the segmenter: SAM2.1 Large with `multimask_output=True` and box-aware mask selection.
- 0 mask fallbacks means the non-empty guarantee is not the bottleneck. Mask quality is: VAL Mask F1 0.666 vs Box F1 0.682.
