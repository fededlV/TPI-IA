## Section 4 — EXP-02C, part 2: anti-collision Weighted Boxes Fusion and sensitivity-group thresholds

**Fusion rule (`src/detection_fusion.py`).**
- Per category, detections are processed in descending calibrated score. Each one joins the cluster whose fused box it overlaps most (IoU ≥ 0.55).
- **Non-self-collision constraint:** a detection may not join a cluster that already holds a box from the *same detector and view* when both are confident (calibrated ≥ 0.5), unless they nearly coincide (IoU ≥ 0.80). Two neighbouring chairs proposed by the same pass must stay two chairs.
- **Fused box:** the score-weighted mean of the members.
- **Fused score:** the sum, over detectors, of each detector's best member score, divided by the number of ensemble detectors.
  - A box seen by both families keeps its full score.
  - A box seen by one family alone is halved. Consensus is promoted without forbidding single-family boxes.

**Ablation (CAL; each arm gets its own CAL-optimal threshold on the fused score).** Each arm changes one ingredient:

| arm | members | change |
|---|---|---|
| C1 | OWLv2 Base | calibrated, NMS (single-family control) |
| C2 | MM-GDINO | calibrated, NMS (single-family control) |
| C3 | both | raw-score union + NMS (no calibration) |
| C4 | both | calibrated union + NMS |
| C5 | both | calibrated WBF without the self-collision rule |
| C6 | both | calibrated WBF **with** the rule |
| C6b / C6c | Large + GDINO / Base + Large + GDINO | C6 with other OWLv2 members |
| C7 | selected C6 arm | + per-group post-fusion thresholds |
