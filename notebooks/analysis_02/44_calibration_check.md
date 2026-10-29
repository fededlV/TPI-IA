**Sanity check: calibration alone does not change a single detector.** The map is monotone and sends τ to 0.5, so cutting the calibrated scores at 0.5 keeps exactly the boxes that pass the raw per-view thresholds.
- **MM-GDINO:** F1 0.6645 → 0.6645.
- **OWLv2 Base:** F1 0.6669 → 0.6635, with the same keep rule.
  - The −0.0034 comes from NMS *ranking* across views: a global box and an overlapping tile box are compared on calibrated scores instead of raw ones, so a different duplicate survives.
  - This was verified offline: NMS ranked by raw score gives exactly 0.6669 again.
  - MM-GDINO is unaffected because both of its views share τ = 0.175.

Choosing one box out of a global/tile duplicate pair is fragile. That is the argument for **fusing** duplicates (WBF) instead of picking one of them (NMS).
