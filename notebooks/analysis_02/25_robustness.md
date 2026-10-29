**Analysis — robustness of the tiling effect across all 8 EXP-02A arms.**

- **Same protocol everywhere.** The B5 protocol was applied unchanged to every backbone × prompting variant: global view at its own optimum, tile threshold tuned, same subset, gate and filter.
- **The tiling gain is robust.**
  - F1 rises in all 8 arms (+0.0055 to +0.0086).
  - Small recall rises in all 8 (+2.8 to +6.7 pp).
  - Large recall is untouched in all 8.
- **The effect size dwarfs the prompting effects.** Tiling adds 6–7 pp of small recall wherever it is applied; the best prompting change (chunking) added 0.5 pp. Resolution, not vocabulary, is what limits small-object detection.
- **Weaker cases.** The gain is smallest for Large with aliases (+2.8 pp small recall, tile threshold 0.30): the extra alias false positives in the tile views force a stricter tile threshold. This is consistent with Section 1: aliases cost precision, and the cost compounds over 4 extra views.
- **Large stays behind Base after tiling** (F1 0.632 vs. 0.667). Tiling does not close the backbone gap.
