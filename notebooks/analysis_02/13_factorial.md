**Analysis — the 2×2 factorial (global view only).**

- **Optimum and curves.**
  - All 8 arms share the same CAL-optimal threshold (0.175) and an almost identical F1 curve; the curves overlap within ±0.01 at every threshold.
  - The prompting variable barely moves the operating point.
- **Chunking (`chunked_raw` vs `single_raw`).**
  - Best global arm: F1 0.6596 vs. 0.6576 (+0.002). Recall +0.65 pp, precision −0.3 pp.
  - The recall ceiling at the 0.05 floor rises from 0.767 to 0.778. This is the one-label-per-block effect: a box whose best overall class was wrong can now also carry the right class from another block.
- **Aliases (`single_alias` vs `single_raw`).** Recall +0.3 pp but precision −1.2 pp (F1 −0.004). Aliases add 1.5 detections per image, and most of them do not match a GT box.
- **Both combined (`chunked_alias`, the master-plan treatment).**
  - Highest raw recall (0.6377) and highest small recall of the global arms (0.523), but also the lowest precision (0.670).
  - Net F1 0.6536, *below* the baseline.
- **OWLv2 Large is worse than Base in every arm.**
  - F1 ≈ 0.625 vs. ≈ 0.656, recall −4 to −5 pp, small recall 0.48 vs. 0.52.
  - Its threshold sweep covers 0.05–0.60 and peaks at the same 0.175, so this is not a mis-chosen operating point.
  - The master plan assumes Large is the stronger detector; this result directly affects T4 (see conclusions).
