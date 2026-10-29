**Scenes.**
- 180 scenes, 6,850 objects, **0 bbox mask fallbacks**. Every SAM2 mask is non-empty, so `area_ratio > 0` holds without the guarantee having to step in.
- The object lists match the frozen EXP-02C detections on all 120 dev images.
- **92% of the submitted objects are cross-family consensus boxes:** 6,299 `wbf:mmgdino+owlv2b`, 377 MM-GDINO only, 174 OWLv2 only.
