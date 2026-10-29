**Analysis — where aliases help and where they hurt (Base, single block, each arm at its CAL optimum).**

- **Net effect.**
  - Over the 45 aliased categories present in CAL (417 GT instances), aliases gain a net **+6 matched instances**, while predictions in those categories grow **+13.6%** (1,082 → 1,229).
  - Each extra true positive costs roughly 24 extra predictions, which is the precision loss measured above.
- **Real wins come from fixing grounding, not archaisms.**
  - `jersey`: 0.667 → 0.905 (+5 of 21 instances) with `sports jersey`.
  - `baseball cap`, `strap` and `handle`: +1 each, matching the EXP-01 note that `strap` grounds poorly with its raw name.
- **The textbook archaisms have no headroom on CAL.**
  - `fireplug`, `spectacles` and `television set` are already at recall 1.0 with their raw names: OWLv2's CLIP text encoder knows them.
  - `fireplug` has only 2 CAL instances, so the motivating example cannot move the metric.
- **A new sibling-like confusion appears.**
  - `computer monitor` falls 0.8 → 0.6 (−2): the `tv`/`television` aliases of `television set` now win the single-block argmax on monitor boxes.
  - This is the failure mode the one-prompt-one-category invariant cannot catch: two *different* aliases that describe the *same* visual concept. The pair (`computer monitor`, `television set`) is a candidate sibling group.
- **Pure false-positive generators.**
  - `tag` triples its predictions (9 → 27) with no recall change; `handle` almost doubles them (48 → 92) for a single extra match.
  - These classes are exactly the "background-prone" categories the master plan flags for stricter thresholds.
- **Decision.**
  - Aliases are **off by default** for OWLv2.
  - They remain available (`detect_variants` always computes them), and they may matter more for MM-GDINO in T4, where the BERT text encoder is less robust to archaic names than CLIP.
