**Analysis — prompt design and disambiguation rules.**

- **Blocks.**
  - Five semantic blocks of exactly 24 classes: apparel, accessories/electronics, kitchen/tableware, home/street fixtures, vehicles/animals/sports.
  - They form an exact partition of ids 1–120 (unit-tested).
  - The grouping follows visual context, so that classes that co-occur and compete for the same boxes are arbitrated against each other.
- **Sibling rule by construction.**
  - Each sibling group (`clock`/`watch`, `trousers`/`jean`/`short pants`, `shoe`/`boot`/`ski boot`, `handle`/`doorknob`/`knob`, `spectacles`/`sunglasses`/`goggles`) lives **inside one block**.
  - Since the label is the per-block argmax, two siblings can never both label the same predicted box. The per-block argmax *is* the disambiguation.
  - A cross-class NMS restricted to sibling groups (`suppress_siblings`, IoU ≥ 0.7) is kept as an explicit second rule for overlapping boxes and measured below.
- **Aliases.**
  - 45 categories carry visual synonyms (183 queries in total). Each alias is remapped to its official `category_id` by max-pooling its scores.
  - Building this table exposed real ambiguities in the inherited alias list, which are now fixed and enforced by a fail-fast invariant (**one prompt → exactly one category**):
    - `"door handle"` was listed under both `doorknob` and `handle`.
    - Bare `glasses` (vs. drinking glass/sunglasses), `t-shirt` (jersey vs. shirt), `watch strap` (strap vs. watch), `mouse` (animal), `cap`, `sign` and `post` were too generic and could remap boxes to the wrong class.
  - An alias that is a synonym of *another* category would silently convert true positives of one class into false positives of another.
