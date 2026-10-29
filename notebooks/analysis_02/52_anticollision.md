**Why the non-self-collision rule does not matter here: a negative result, explained.**
- The rule protects adjacent instances that come from the *same* detector view. But every view is already de-duplicated upstream: per-class NMS at IoU 0.5 inside each detector pass.
- Of the **64,944** detections entering fusion on DEV, **zero** same-view same-class pairs reach IoU ≥ 0.55. The WBF entry condition can never be met by two such boxes against each other.
- The rule can only fire indirectly: when the *fused* box of a cluster drifts towards a neighbouring same-view box because of another detector's member.
- Measured on DEV, that drift case changes the fusion on 12/120 images, by 11 boxes before thresholding and **3 boxes at the operating point**. No GT instance changes its match status.

**Conclusion.** The collapse of adjacent instances that the rule guards against does not happen in this pipeline, because the per-view NMS at 0.5 already prevents it. The rule is kept as a cheap invariant: it is unit-tested, and it would matter if the per-view NMS were relaxed or removed. It is not a source of the EXP-02C gain, and no gain is attributed to it.
