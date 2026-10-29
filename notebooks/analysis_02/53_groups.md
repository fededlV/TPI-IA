**Sensitivity-group thresholds (fitted on CAL, frozen to `artifacts/cache_exp02/exp02c/config.json`).**

*The groups.*
- **micro:** 57 categories with CAL GT median `area_ratio` < 0.0035, covering 643 CAL instances.
- **standard:** the other 63 categories, covering 588 instances.
- **fp_prone:** **empty.** No category produced ≥ 3 verified false positives with ≥ 50% of its evaluated predictions at the operating point. Verified-negative FPs are rare under consensus fusion: only 3 of 496 EXP-02C box FPs on DEV, about 0.6%.
- The master plan's intuition (`strap`, `tag`, `pole`, `handle`, `button`) is therefore not supported by this data at this operating point. The group stays defined but empty, rather than being filled by hand.

*Result.*
- Coordinate ascent converges in one round: standard 0.450 and micro **0.425**. Micro-objects get a slightly more permissive threshold, as the LVIS recall argument predicts.
- C7 vs C6: recall +1.5 pp and R_small **+2.4 pp** (significant), precision −0.8 pp. Net **ΔF1 +0.0038 [−0.002, +0.010]**, not significant.
- It costs 2 extra fitted parameters, so the group thresholds are a small refinement, not a driver.

*Integrity check.* The frozen JSON config, replayed through the blind inference path (`fuse_image`), reproduces the tuned CAL detections exactly.
