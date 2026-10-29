## Section 5 — Submission #2: frozen EXP-02C boxes + SAM2.1, and the contract suite

**Pipeline.**
1. The frozen config (`exp02c/config.json`, sha `912f09d78ec5dd6e`) is applied blind to all 180 images: tile filter → per-view calibration → anti-collision WBF → group thresholds → top-300.
2. `scripts/run_exp02c_segmentation.py` segments every box with **SAM2.1 Base+, `multimask_output=False`**. This is the same segmenter as EXP-01, so any Mask F1 change comes from the boxes alone. Multimask selection and SAM2.1 Large belong to EXP-03 (T5).
3. The scenes are written resumably to `exp02c/scenes.jsonl`. The writer refuses to mix scenes from another config or segmenter.
