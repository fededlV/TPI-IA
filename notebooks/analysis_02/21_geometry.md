**Analysis — tile geometry.**

- **Tile size and overlap.**
  - For a 640×480 image (`tpi_0100`, the CAL image with the most small GT instances: 43), `tile_grid` yields four 366×275 windows.
  - The shared bands are 92 px horizontally and 70 px vertically: 25.1% and 25.5% of the tile side. *25% overlap* is defined relative to the tile, hence `tile = dim / 1.75`.
- **Coverage guarantee (unit-tested).**
  - Any object narrower than the horizontal band and shorter than the vertical band (minus the 3-px margin, i.e. up to ~85 × 63 px here) lies entirely inside at least one tile **without touching that tile's inner border**.
  - This is what makes the truncation filter safe. Every box that touches an inner border is either a partial view of an object that another tile (or the global view) sees whole, or an object too large for tiling to matter.
- **Inner vs. image borders.** Only *inner* borders count as truncation. A tile border that coincides with the image border is the photo's own crop, and objects cut there are cut in the global view too, so filtering them would only lose recall.
- **Projection.** Tile boxes are translated by the tile origin, clipped to `[0, W] × [0, H]`, and tagged `owlv2:tile{i}`, so every merged detection keeps its view of origin for later traceability and calibration.
