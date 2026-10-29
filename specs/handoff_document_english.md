# Handoff: TPI Visual Scene Understanding (`tpi-clean-slate`)

- **From:** Antigravity (Specification & Architecture Agent)
- **To:** Claude Code (Build Agent)
- **Date:** 2026-10-28
- **Spec Status:** `Approved` (Approved by Fede on 2026-10-28 — Group 2: `TPI_IA_2026P_2.ipynb`)

---

## 1. Key Architectural & Normative Findings to Respect During Build

1. **Dataset Location:** `/home/fede/Ingenieria_en_Sistemas/Inteligencia Artificial/Tp/dataset/`
   - `development/annotations.json`: 120 images (`tpi_0001.jpg`..`tpi_0120.jpg`), 120 categories, 1,790 GT instances. Uses `xywh` bounding boxes and integer `image_id` (`1`..`120`).
   - `tpi_ia_2026_competition_public/`: 60 test images (`test_0001.jpg`..`test_0060.jpg`), `categories.csv`, `test_images.csv`.
   - `tpi_2026_qa_public/questions_dev.jsonl`: 60 entries across 20 dev images (3 per image: 1 `description` + 2 closed questions).
   - `scene_schema.json` & `sample_submission.csv`: Normative 80-row submission schema (`60` test rows + `20` dev benchmark rows).
2. **Critical Schema Traps Fixed in Master Plan:**
   - Field names in `SceneJSON`: **`area_ratio`** (NOT `relative_area`) and **`relations`** (NOT `spatial_relations`).
   - `area_ratio` has `"exclusiveMinimum": 0` in `scene_schema.json` — every object mask MUST have $\ge 1$ active pixel (`area_ratio > 0`).
   - Bounding boxes in `SceneJSON` must be **`[x1, y1, x2, y2]` (`xyxy`)**.
   - In `questions` and `questions_llm`, `normalized_answer` for `type == "count"` must be a Python **`int`** (`0, 1, 2...`), never a string `"0"`.
   - Both `questions` and `questions_llm` must include all 3 items per dev image (including `type == "description"`).
   - In `questions_dev.jsonl`, `type == "relation"` questions do not have a separate `predicate` field; extract the predicate (`above`, `below`, `left_of`, `right_of`, `overlaps`) deterministically from the Spanish `question` text.
   - `SceneJSON["relations"]` must compute full pairwise `left_of`/`right_of`/`above`/`below` by pure centroid comparison (`cx_A < cx_B`, `cy_A < cy_B`) without perpendicular-axis overlap filtering (required by pairs like `DEV-5-Q2` `clock` vs `ring` and `DEV-15-Q1` `bottle` vs `strap`).

---

## 2. Execution Order for Claude Code

Execute tasks `T1` through `T8` sequentially using the exact prompts and `/effort` levels documented in Section 8 of `Plan Maestro Integral desde Cero — TPI Comprensión Visual de Escenas (Primavera 2026).md`, clearing context (`/clear`) between tasks.
---

## 3. T1 Build Log (Claude Code — 2026-10-28, branch `feat/t1-contract-foundations`)

### Spec-Challenge Notes (T1 / S1)
1. **Layout (non-blocking):** specs live as flat files (`specs/*_english.md`), not `specs/tpi-clean-slate/{spec,tasks,handoff}.md`; used as-is.
2. **No `AGENTS.md` (non-blocking):** test runner taken from T1 verification command: `python3 -m unittest discover -s tests -p "test_contracts.py"`.
3. **No `pos_category_ids` in `annotations.json` (non-blocking):** positive categories are derived from categories with GT annotations per image (dataset README semantics). 36 images have `not_exhaustive_category_ids`, 74 have `neg_category_ids`.
4. **GT segmentation = polygons; no `pycocotools`/`cv2`/`skimage`/`torch` locally (non-blocking):** polygons rasterized with `PIL.ImageDraw`; erosion via `scipy.ndimage`; LAB via numpy sRGB->XYZ->LAB; torch is optional (guarded import).
5. **Suite vs. placeholder sample (resolved by human):** `run_full_verification_suite(..., require_qa=True, infer_fn=None, qa_fn=None, vram_peak_mb=None)`. `sample_submission.csv` has empty `questions`, so it is checked with `require_qa=False`. Tests 1 and 8 report `SKIPPED` and Test 2 reports `PARTIAL` (QA sub-check skipped) until T2+ supplies the hooks.
6. **Centroid units (non-blocking):** absolute pixel indices `[mean(x), mean(y)]` (matches `sample_submission.csv`: 1px mask at (0,0) -> `[0, 0]`). Position = `floor(3*cx/W)`, `floor(3*cy/H)` clamped to [0, 2].
7. **Relation cap (finding):** top-95 objects alone do NOT guarantee `< 20000` relations (worst case 95*94*3 = 26,790). `build_relations` additionally hard-caps at 19,998, emitting symmetric pairs atomically and highest-confidence pairs first.
8. **Test 1 (blind inference):** implemented with a `sys.addaudithook` guard detecting any `open()` of `annotations.json` instead of renaming the official file (non-destructive, works on read-only Kaggle input).
9. **Test 6** additionally verifies derived-geometry consistency (area_ratio, centroid, position, size vs. decoded mask) and valid `xyxy` bbox inside the image (feeds the 15% JSON Consistency term).

### Results
- `python3 -m unittest discover -s tests -p "test_contracts.py"` -> 32 tests OK.
- `python3 scripts/build_gt_oracle.py` -> 20 GT scenes, 230 objects, 5,710 relations, 0 mask fallbacks, 0 schema errors; Box F1 = 1.0, Mask F1 = 1.0, coverage = 1.0; suite on `sample_submission.csv` and on sample+GT-oracle rows: tests 3/4/5/6/7 PASS, 2 PARTIAL, 1/8 SKIPPED, 0 FAIL. **S1 ORACLE: PASS.**
- `artifacts/split_cal_val.json`: CAL 80 (10 QA), VAL 40 (10 QA), SEED 42, non-QA images stratified by instance-count tercile.
- Dataset root auto-resolved locally to `/home/fede/Ingenieria_en_Sistemas/Inteligencia Artificial/Tp/dataset` (override with `TPI_DATASET_DIR`).

### Notes for T2
- Pass `infer_fn(image_path) -> SceneJSON`, `qa_fn(scene, questions) -> answers` and `vram_peak_mb` to `run_full_verification_suite` to activate Tests 1, 2 (QA part) and 8.
- The GT oracle F1 = 1.0 is self-consistent (same rasterizer on both sides); federated TP/FP/FN/ignore logic is covered by synthetic unit tests.

---

## 4. T2 Build Log (Claude Code — 2026-10-28, branch `feat/t2-qa-engine-baseline`)

### Spec-Challenge Notes (T2 / S2)
1. **Compute (resolved by human: local CPU venv):** no CUDA device on the workstation (Intel UHD, 4 cores, 14 GB RAM; system Python 3.14). A `uv` venv (Python 3.12, torch CPU) pinned in `requirements.txt` runs EXP-01; inference is cached under `artifacts/cache_exp01/` (~75 CPU-minutes for 180 images).
2. **`Box F1 >= 0.45` is not guaranteed by construction** with OWLv2 Base + raw names; EXP-01 keeps aliases/chunking/tiling out (they are the EXP-02 variables). Only one global score threshold is tuned, on CAL; VAL is reported, never used for selection.
3. **Test 8 (VRAM) on CPU:** `vram_peak_mb=None` → `SKIPPED` (never a fake `0`). "All 8 tests" = 0 FAIL with Test 8 SKIPPED until the Kaggle GPU run.
4. **Relation semantics:** "any instance pair" as instructed; evidence = highest-confidence qualifying pair. GT has exactly one instance per class in all 10 relation questions, so GT verification is unambiguous; on predictions any-pair can say `yes` to a predicate and its inverse under FPs (top-confidence-pair is the T3+ alternative).
5. **Missing class:** `relation`/`existence` → `"no"` (allowed answers are yes/no), `evidence=[]`; `position`/`size` → `"insufficient_information"` (prompt + suite Test 2).
6. **`questions_llm` before the SLM (Prohibition 4):** `categories.csv` is English-only, so a generic 120-category Spanish lexicon (`CATEGORY_NAMES_ES`, validated for coverage and ambiguity) was added to `config_and_seeds.py`. A regex interpreter reads only the raw question text + vocabulary, emits `interpreted_query` (`source: rule_based_fallback`), and delegates to the deterministic engine.
7. **No gold QA file:** "validate the 60 answers" = GT-derived expectations pinned in `tests/test_qa_engine.py` (counts from annotations, existence POS/NEG, position from GT centroids, size by thresholds, relations from GT centroids; 5 yes / 5 no).
8. **EDA definition note:** 71.06% < 1% is reproduced from the COCO `area` field; PIL polygon rasterisation gives 70.34% (boundary pixels). Both are reported in the notebook.
9. No `AGENTS.md`: runner `python -m unittest discover -s tests` (via `.venv/bin/python`).

### Results (T2 / S2)
- Unit tests: `.venv/bin/python -m unittest discover -s tests` → 60 OK (T1 contracts + QA engine + perception/pipeline, model-free).
- QA engine on GT scenes: 40/40 closed answers match GT-derived expectations; rule-based `questions_llm` agrees 40/40.
- EXP-01 (OWLv2 Base, raw names, threshold **0.175** = argmax CAL Box F1; SAM2.1 Base+ `multimask_output=False`):
  - CAL Box F1 0.6576 / Mask F1 0.6400; VAL Box F1 0.6389 / Mask F1 0.6198; **DEV-120 Box F1 0.6518 / Mask F1 0.6338**; LVIS coverage 0.46; 0 bbox mask fallbacks over 5,592 objects.
  - Box recall by size: small 0.499 / medium 0.742 / large 0.787 (~69% of FNs are small objects).
  - Deterministic QA on predicted scenes: 30/40 (count 4/10, size 3/5, existence 9/10, relation 9/10, position 5/5); all misses are perception-propagated.
  - CPU latency ≈ 21.6 s/img (11.9 det + 9.5 seg + 0.24 build); VRAM N/A (CPU).
- `submissions/submission_exp01_baseline.csv` (80 rows): suite 7 PASS / Test 8 SKIPPED (no GPU) / **0 FAIL**.
- Fresh-vs-cache OWLv2 check on `tpi_0003` and `test_0007`: bit-exact.

### Notes for T3
- Raw OWLv2 Base detections (score ≥ 0.05) for all 180 images are in `artifacts/cache_exp01/detections.json` (committed) — reuse as the EXP-01 reference in EXP-02A/B ablations without re-running.
- `scenes_thr0.175.jsonl` (39 MB) is git-ignored; regenerate with `python scripts/run_exp01_inference.py segment --threshold 0.175`.
- Run T3+ on a GPU (Kaggle recommended): models auto-select CUDA via `config_and_seeds.resolve_device()`; see `notebooks/README.md`. With CUDA, the notebook passes the traced VRAM peak to suite Test 8.
- Biggest levers measured: small-object recall (tiling), grounding of archaic names (`strap`, `cabinet`; also `elephant` raw 0.088), nested duplicates / over-split instances (`flowerpot`, `pillow`, `handbag`) hurting counts.
