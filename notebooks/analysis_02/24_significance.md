**Analysis — statistical significance and the S3 verdict.**

- **B5 vs. B0 (the effect of selective tiling alone).**
  - Recall +3.8 pp, CI [+2.7, +4.9].
  - **Small recall +6.4 pp, CI [+4.6, +8.2].**
  - Medium recall +1.4 pp, CI [+0.2, +2.7].
  - Large recall exactly unchanged: by construction, the size gate and the fixed global threshold make tiling unable to touch large objects.
  - These gains are paid for with precision −2.5 pp, CI [−3.8, −1.4]. Net Box F1 **+0.0073, CI [−0.0030, +0.0177], p(Δ ≤ 0) = 0.09: not significant.**
- **B5 vs. EXP-01 (the full EXP-02A+B system against the baseline; this is the S3 criterion).**
  - **Small-object recall (`area_ratio < 0.0035`): 0.516 → 0.585, +6.9 pp, CI [+5.1, +8.7], p < 0.001: significant ✔.**
  - Medium recall +2.1 pp (significant). Large +1.2 pp (not significant).
  - **Global Box F1: 0.6576 → 0.6669, +0.0093, CI [−0.0024, +0.0210], p(Δ ≤ 0) = 0.06: positive, but *not* significant at 95% ✘.**
- **S3 verdict: partially met.**
  - The small-object recall criterion is clearly met.
  - The global Box F1 criterion is met in direction, but its 95% CI includes 0 on 80 CAL images.
  - Both numbers are in-sample (thresholds tuned on CAL), so the true effect is if anything smaller. The held-out confirmation is T4's VAL comparison.
- **Why F1 lags behind recall.**
  - Under the federated LVIS policy every extra tile detection of a verified-negative or exhaustive-positive category is a false positive. Tiles find small objects, but they also find small *non-objects*, and they are not corroborated by the global view.
  - With independent per-view scores and NMS-only merging, the recall gain is largely cancelled by precision. That points at T4's tools: score calibration per view/detector and fusion (WBF / cross-view consensus), so that a box seen in two views is promoted and an isolated tile box is demoted.
- **B2 vs. B0 is the cautionary control.** The shared-threshold implementation *significantly loses* 8.1 pp of large recall (CI [−13.1, −3.4]) for no significant F1 gain. A naive tiling implementation would have reported a misleading "small-object win" while silently degrading large objects.
