**Analysis 3 — scene density and intra-class overlap (Plot 3a/3b).**

- **Density:** 5 to 66 instances per image (median 10.5, mean 14.9, right-skewed); 5 images contain ≥ 50 instances. The 300-object cap of the schema is never binding for GT, but an over-permissive detector on dense scenes could approach the 95-object relation cap — another reason to keep the threshold calibrated.
- **Intra-class overlap is rare but real:** among 3,836 same-class GT box pairs, 11.6% touch (IoU > 0), only 1.07% have IoU ≥ 0.3 and **19 pairs (0.50%) have IoU ≥ 0.5** (15 at ≥ 0.55; maximum 0.775). The classes that most often appear multiple times per image are small, repeated items: `sock`, `pillow`, `glove`, `drawer`, `chair`, `person`, `napkin`, `strap`.

**Decisions.**
1. **Per-class NMS at IoU 0.5** (EXP-01) can suppress at most those 19 genuinely overlapping GT pairs (≈1% of instances) while removing the far more frequent duplicate detections — an acceptable trade-off for the baseline.
2. **WBF anti-collision rule (EXP-02C):** fusing boxes of the same class from the *same* detector/view at IoU ≥ 0.55 would merge 15 real pairs in development (paired socks, gloves, stacked pillows). The plan's constraint — only fuse same-source boxes at IoU ≥ 0.80 — would merge none of them (max observed 0.775). This number is the quantitative justification of that threshold.
3. Mask-level reasoning (EXP-05 collective suppression) is needed for group boxes, because box IoU alone cannot separate a "group" prediction from its members.
