**MM-GDINO alone is as good as OWLv2 Base, but it is good at different things.**
- With selective tiling: MM-GDINO F1 **0.6645** vs OWLv2 Base **0.6669**. ΔF1 = −0.0024, 95% CI [−0.028, +0.025]: no difference.
- The profiles differ:
  - MM-GDINO is far more precise on the full image (P 0.729 vs 0.685) and much stronger on large objects (R_large 0.881 vs 0.800).
  - It is weaker on small objects (full-image R_small 0.419 vs 0.521).
- Tiling helps MM-GDINO even more than OWLv2: small recall +12.7 pp (0.419 → 0.546).

**Complementarity (CAL GT instances matched by each family at its own operating point).**
- Small objects: 279 by both, **95 by OWLv2 only, 70 by MM-GDINO only**, 195 by neither. The union of the two families could reach **0.695** small recall, against 0.585 for the best single family.
- Medium: oracle union recall 0.852. Large: 0.938.
- This is the diversity the ensemble needs: each family recovers objects the other misses, in every size bucket.

**Selected:** `chunked_raw`, global threshold 0.175, tile threshold 0.175. Only the raw-name cache was generated (`RUN_ALIAS = False`), so the alias re-test for GDINO remains open.
