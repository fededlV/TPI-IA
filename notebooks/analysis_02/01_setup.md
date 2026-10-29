**Analysis — provenance and evaluation protocol.**

- **Provenance.**
  - Both backbones ran on the same Tesla T4 (torch 2.11.0+cu128), over all 180 images (120 dev + 60 test, so T4 can reuse the cache).
  - The same 183 text queries were used for both: 120 official names + 63 aliases.
  - The score floor is 0.05, as in EXP-01.
  - Tile detections were cached **before** truncation filtering (`truncation-filtered in cache = False`), so the inner-edge filter, the tiled-category subset and the tile threshold can all be ablated here without touching the GPU again.
- **Where the errors are.**
  - CAL contains 639 small, 432 medium and 160 large GT instances: **51.9% of the instances to be found are small**.
  - At EXP-01's small recall (0.516), this bucket alone accounts for ~309 missed boxes. That is the target of this notebook.
- **Why box-only.**
  - The score threshold decides *which* objects exist. SAM2 only decides their shape (EXP-03), so detection choices are scored on boxes and segmentation stays out of the decision, as in EXP-01.
  - `per_image_counts` stores TP/FP/FN and matched/total GT per size bucket for each image. Bootstrap resamples images (the unit of variation) and recomputes F1 from the summed counts, exactly as the official metric aggregates them.
