**Analysis — paired bootstrap of the EXP-02A effects (95% CI over CAL images).**

- **No prompting variant changes Box F1 significantly.**
  - Every F1 delta has a CI that straddles 0, with point estimates between −0.004 and +0.002.
- **Chunking buys recall at no significant precision cost.**
  - Base: R +0.65 pp, CI [+0.17, +1.34], p = 0.002. Precision −0.32 pp, CI [−0.79, +0.15], not significant.
  - Large replicates it: R +0.41 pp, CI [+0.08, +0.81]. The effect is small but real and consistent with the mechanism (extra labels per box).
- **Aliases significantly *hurt* precision and do not significantly help recall.**
  - Base `single_alias`: P −1.21 pp, CI [−2.33, −0.23]. Base `chunked_alias`: P −1.79 pp, CI [−2.94, −0.72], p(Δ ≤ 0) = 1.00.
  - Large shows the same pattern.
  - Small-recall gains from aliases (+0.3 to +0.6 pp) are not significant.
- **Large vs. Base (`single_raw`).**
  - F1 −0.032, CI [−0.056, −0.008]: significant. Recall −4.4 pp: significant. Small recall −4.1 pp, with the CI touching 0.
- **Verdict on EXP-02A.**
  - The "5 blocks + aliases" hypothesis is **refuted for OWLv2 as a Box-F1 lever**. Chunking is a mild, significant recall gain at no inference cost; aliases are a net precision loss.
  - The architecture predicted this outcome before any data was seen. OWLv2's CLIP text tower embeds each query independently, so there was never any attention saturation to remove.
