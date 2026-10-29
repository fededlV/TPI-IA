**Analysis 1 — object scale (Plot 1a/1b).**

- **Headline numbers:** **71.06%** of the 1,790 GT instances occupy less than 1% of the image and the median `area_ratio` is **0.00289** (≈ 0.29% of the image — roughly a 28×28 px patch in a 640×427 photo). P10 = 0.00018 means one instance in ten is smaller than ~7×7 px.
- **Definition matters (and we report both):** the 71.06% figure is reproduced exactly from the COCO `area` field of `annotations.json`. Rasterising the polygons with PIL gives 70.34% / median 0.00316 because the rasteriser also fills boundary pixels — a +9% relative area inflation at the median that matters for tiny objects. Since the metric scores *our* masks against *its* rasterisation, this is a reminder that micro-object Mask IoU is fragile: a 1-pixel boundary disagreement on a 7×7 object already costs 25–40% IoU (7×7 vs 6×6 → IoU 0.73; 7×7 vs 9×9 → IoU 0.60).
- **Size buckets under the canonical thresholds** (`small < 0.0035 ≤ medium < 0.035 ≤ large`): 53.2% small, 33.7% medium, 13.0% large. More than half of the recall budget therefore lives in objects smaller than a 32×32 COCO box.

**Decisions.**
1. OWLv2 runs at its native 960×960 resolution (no downscaling); EXP-02B adds a 2×2 tiled pass with 25% overlap so that small objects are seen at ~2× effective resolution.
2. Recall is broken down by size bucket in Section 3 so that EXP-02B's effect can be attributed to the `small` bucket specifically.
3. The `size` thresholds stay at COCO-relative values (0.0035 / 0.035), which classify the five benchmark `size` objects consistently (Plot 4c).
