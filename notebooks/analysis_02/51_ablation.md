**Where the gain comes from (CAL, paired bootstrap).**

1. **The second family is the big lever.** The C6 ensemble beats OWLv2 alone (C1) by **ΔF1 +0.058 [+0.034, +0.079]**, with precision +7.4 pp and recall +4.3 pp, both significant. Even the naive raw union (C3, F1 0.7057) beats either family alone (0.665–0.670).

2. **Calibration is neutral under NMS.** C4 vs C3 gives ΔF1 +0.0014 (n.s.).
   - Calibration trades recall for precision (+2.5 pp P, −1.7 pp R).
   - Because the raw F1-optimal thresholds already coincide (τ = 0.175), the two raw scales are less mismatched than the master plan assumed.
   - Calibration earns its place inside WBF, where the fused score *adds* the two families' scores. On raw scales an MM-GDINO box would contribute more to the sum than an OWLv2 box of the same rank (22% vs 7% of detections above 0.3), so the sum would not weigh the two families equally.

3. **WBF beats NMS on recall.** C6 vs C4 gives ΔF1 +0.016 [−0.004, +0.035], p = 0.051, just short of significance at 95%. The recall gain is significant: **R +4.8 pp, R_small +5.8 pp**.
   - NMS can only keep one of two agreeing boxes. WBF keeps the consensus box *and* rewards it with a higher fused score, which allows a lower operating threshold (0.45 vs 0.75).

4. **The OWLv2 member.**
   - Base + MM-GDINO (C6, F1 0.7236) beats Large + MM-GDINO (C6b, 0.7081).
   - Adding Large as a third member (C6c, 0.7234) brings nothing: two OWLv2 backbones are the same family and add no diversity.
   - **Selected: OWLv2 Base + MM-GDINO** (max CAL F1, fewer members on ties). This answers Spec-Challenge Note 1: the "OWLv2 Large" named in S4 is not the better member on this data.

5. **The self-collision rule has no measurable effect.** C5 and C6 are identical (ΔF1 = 0.0000). See the next cell.

Fusion costs 14 ms/img on CPU.
