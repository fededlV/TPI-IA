**Analysis — computational cost (Kaggle Tesla T4, FP32).**

- **Base.**
  - 2.65 s per image for the 5 views (≈ 0.53 s per view), p95 2.79 s.
  - Peak VRAM 780 MB.
  - The timing includes preprocessing, the 183-query forward and the NumPy extraction of 4 variants with per-class NMS.
- **Large.**
  - 11.4 s per image (≈ 2.3 s per view): 4.3× Base, consistent with a ViT-L/14 at 1008 px vs. a ViT-B/16 at 960 px.
  - Peak VRAM 2.0 GB. Both are far below the 14 GB budget of suite Test 8.
- **Where the cost comes from.**
  - Chunking and aliases are **free** with OWLv2: one forward pass, 183 instead of 120 short text queries.
  - Tiling multiplies the detection cost by **5×** (1 global + 4 tiles).
  - For the 60 test images, Base + tiling costs ~2.7 minutes on a T4, which is acceptable for the submission. Large + tiling (~11 minutes) is hard to justify given that it is the worse detector.
- **Cache size.** Tile views produce ~1.9× as many raw candidates as the global view (335 vs. 179 per image for Base `chunked_alias` at the 0.05 floor). Compressed, all 4 variants × 5 views for 180 images fit in ~7 MB per backbone.
