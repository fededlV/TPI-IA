**Box and mask quality vs EXP-01** (paired bootstrap; both arms use SAM2.1 Base+ with one mask per box).

| split | metric | EXP-01 | EXP-02C | Δ [95% CI] |
|---|---|---|---|---|
| CAL | Box F1 | 0.6576 | 0.7273 | +0.0697 [+0.042, +0.093] |
| CAL | Mask F1 | 0.6400 | 0.7047 | +0.0647 [+0.036, +0.089] |
| VAL | Box F1 | 0.6389 | 0.6822 | **+0.0434 [+0.014, +0.079]** |
| VAL | Mask F1 | 0.6198 | 0.6655 | **+0.0457 [+0.018, +0.083]** |
| VAL | 0.40·Box + 0.35·Mask | 0.4725 | 0.5058 | **+0.0333 [+0.013, +0.059]** |

**Reading the table.**
- The box gain carries over almost one-to-one to masks, because the segmenter is unchanged. A better box prompt gives a better mask.
- On the perception part of the `PipelineScore` (75% of the total weight), the held-out gain is **+3.3 points**, significant.
- The EXP-01 scenes were re-segmented on the same GPU for this comparison. They reproduce the T2 CPU figures exactly (CAL Mask F1 0.6400, VAL 0.6198).

**Cost per image (Tesla T4, FP32, models run one after another).**

| stage | time |
|---|---|
| OWLv2 Base (5 views) | 2.6 s |
| MM-GDINO Large (25 forwards) | **62.7 s** (n = 150 traced images) |
| fusion | 0.014 s (CPU) |
| SAM2.1 Base+ | 0.54 s |
| scene build | 0.33 s |
| **total** | **≈ 66 s** |

- Peak VRAM is **8.3 GB**, within the 14 GB budget.
- MM-GDINO is 95% of the latency. It is the obvious target if inference time matters: FP16, or batching the 5 blocks of a view into one forward.
