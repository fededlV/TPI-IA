**Analysis — `run_full_verification_suite` on Submission #1: 7 PASS, 1 SKIPPED, 0 FAIL.**

| Test | Result | What it proves for this submission |
|---|---|---|
| 1 Blind inference | PASS | The full pipeline (`OWLv2 → SAM2 → SceneJSON`) ran on a synthetic external image while an audit hook watched every `open()`: `annotations.json` was never touched and the output validated against the schema. |
| 2 Edge cases | PASS | Empty scene validates; QA on it returns `0` (int), `no`, `insufficient_information` ×2 with empty evidence and a non-empty description; a 350-object scene is truncated to 300 objects / < 20,000 relations in < 2 s. |
| 3 Relation symmetry | PASS | For all 80 rows, every `left_of/above/overlaps` triple has its inverse and only references existing objects. |
| 4 QA integrity | PASS | 3 + 3 prompts per dev row, `count` answers are `int`, evidence ids exist, are unique, and are empty for `0` / `no` / `insufficient_information`. |
| 5 Schema | PASS | `jsonschema` (Draft 2020-12) on all 80 rows: 0 errors. |
| 6 RLE integrity | PASS | `size == [H, W]`, counts sum to `H·W`, non-empty decoded masks, `0 < area_ratio ≤ 1`, and centroid/area/position/size re-derived from each mask match the stored values. |
| 7 CSV structure | PASS | Exact columns, 80 rows in sample order, parseable JSON, image ids/dimensions match the sample. |
| 8 VRAM budget | SKIPPED | No GPU on this machine; we refuse to report a fabricated value. It will be activated by the Kaggle T4 run. |

The contract part of S2 is therefore satisfied with **0 failures**; the 15% JSON-consistency term is protected by construction rather than by luck.
