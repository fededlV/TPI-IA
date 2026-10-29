**Analysis — regression check against EXP-01.**

- **The control arm is the baseline.** `single_raw` from the 183-query alias-union pass on the T4 GPU reproduces the EXP-01 cache (120-query pass on CPU) **exactly at the metric level**:
  - same P / R / F1 (0.6883 / 0.6296 / 0.6576);
  - same recall in every size bucket;
  - same 33.1 detections per image;
  - identical detection counts in 80/80 images.
- **Residual differences are numerical, not behavioural.**
  - The maximum score difference is 7·10⁻⁴: GPU vs. CPU float arithmetic plus the 4-decimal rounding of the compressed cache.
  - The 37 detections "without a counterpart" are an artifact of this check's own key, which rounds coordinates to 0.1 px. A coordinate that moved by 10⁻⁴ across a rounding boundary changes the key. Counts and all metrics are identical, so no detection actually appeared or disappeared.
- **Consequence.** Adding 63 alias queries does not perturb the official-name scores (as the architecture predicts), and the GPU run is a faithful superset of EXP-01. Every delta below can be attributed to the manipulated variable and not to the environment.
