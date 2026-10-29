**Analysis — what tiling rescues.**

- **Overall.**
  - Across CAL, the final configuration matches **50 GT instances** that the full-image view missed (42 small, 8 medium) and loses only 3 (1 small, 2 medium).
  - A rescue/loss ratio of ~17:1 is the per-instance picture behind the +6.4 pp small recall.
- **The 3 losses.** A higher-scoring tile box of the same class wins per-class NMS over a better-localised global box, then falls below IoU 0.5 with the GT. Cross-view WBF (T4) would average the two boxes instead of picking one.
- **Rescued categories are the micro-objects that motivated the experiment:** `knife` (5), `earring` (3), `pillow` (3), `book`, `awning`, `napkin`, `license plate`, `sock`, `flowerpot`, `taillight` (2 each).
- **Showcase image `tpi_0115` (a kitchen).**
  - All 5 rescues are knives hanging on wall racks, each with `area_ratio` 0.00018–0.00031 (≈ 55–95 px² in a 640×480 image).
  - At the global resolution each knife spans about one OWLv2 patch; in the tile it spans several.
  - The objects still missed (red) are partly occluded pots and utensils clustered at the top of the racks. Higher resolution alone does not separate touching instances; that requires better localisation or a different detector family.
