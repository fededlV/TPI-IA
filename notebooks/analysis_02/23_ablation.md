**Analysis — the incremental tiling ablation (CAL, Base `chunked_raw`).**

- **B1, naive tiling, is harmful: F1 0.660 → 0.607.**
  - Without the truncation filter, every object crossing an inner border produces partial boxes; precision collapses 0.685 → 0.595.
  - The filter removes **27.8%** of all tile detections above the final tile threshold (826 of 2,968). Truncation artifacts are not an edge case; they are a quarter of what tiling produces.
- **B2, filter only, still hides a trap.**
  - F1 recovers to 0.663 and small recall climbs to 0.567, but **large recall falls 0.800 → 0.719**.
  - With a single threshold shared by all views, the optimum moves from 0.175 to 0.225 to absorb the extra tile false positives. Global detections of large objects scoring 0.175–0.225 are lost, even though tiling never touched them.
  - The tile views and the global view have **different score scales** (the zoomed-in view is more confident on small things and noisier overall), and one threshold cannot serve both.
- **B3, per-view thresholds, fixes it.**
  - The global view stays at 0.175 and tiles get their own threshold (0.275).
  - Large recall returns to 0.806, recall rises to 0.663 and F1 to 0.6645.
- **B4, size gate:** a small precision gain (0.666 → 0.668) for the same recall. Tiles stop contributing large boxes, which they could only produce as partial views anyway.
- **B5, category subset, gives the best arm.**
  - Excluding the 14 large categories (`person` in particular) removes a major source of tile false positives. The CAL-optimal tile threshold can then drop to 0.25, which lifts small recall to **0.585** and medium recall to 0.759.
  - Final CAL Box F1 0.6669, recall 0.674 (+3.8 pp over B0), precision 0.660 (−2.5 pp).
- **Reading.** Each ingredient is necessary. The naive version of the technique (B1) is *worse than not tiling at all*, which is why the truncation filter and per-view thresholds are part of the method and not optional refinements.
