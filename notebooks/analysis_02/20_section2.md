## Section 2 — EXP-02B: Full Image vs. Full Image + Selective 2×2 Tiling

**Mechanism.**
- OWLv2 resizes every input to a fixed square (960 px for Base).
- A 640×480 photo is upsampled 1.5× and each 16-px patch covers ~10.7 px of the original. A ring or a watch of 10–15 px therefore spans one or two patches: below what the box head can localise.
- Cropping a 366×275 tile and feeding it to the same model upsamples it 2.6×, so each patch covers ~6.1 px: **1.75× more resolution per axis** on the same object.

**Arms.**
- Each arm adds one ingredient to the previous one, and its free thresholds are re-tuned on CAL:

| arm | change |
|---|---|
| B0 | full image only (EXP-02A winner) |
| B1 | + 4 tiles, one shared threshold, no truncation filter (the naive implementation) |
| B2 | + inner-edge truncation filter |
| B3 | + per-view thresholds (global fixed at its B0 optimum, tile threshold tuned) |
| B4 | + size gate (tiles only contribute boxes with `area_ratio < 0.035`) |
| B5 | + category subset: only small/medium categories are tiled (**selective tiling**, final) |

- All views are merged by per-class NMS (IoU 0.5). Score fusion across views (WBF) is deliberately left to T4.
