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

---

## 5. T3 Build Log (Claude Code — 2026-10-29, branch `feat/t3-perception-chunk-tiling`)

### Spec-Challenge Notes (T3 / S3)
1. **OWLv2 has no text "attention dilution" (non-blocking):** each query gets an independent logit (`modeling_owlv2.py`: per-query einsum; shift/scale depend on the image only). For OWLv2, chunking only changes the per-box label assignment: argmax within each block instead of across all 120 classes (up to 5 labels per box). One forward pass with the union of alias queries is mathematically identical to 5 chunked passes, so the 2x2 factorial (raw/alias x 1/5 blocks) costs a single image encoding. The token-saturation argument applies to GDINO-family models (256-token limit) in T4.
2. **Alias collision (fixed in `config_and_seeds.py`):** `"door handle"` was listed under both `doorknob` and `handle`; ambiguous aliases (`glasses`, `t-shirt`, `watch strap`, bare `mouse`, `cap`, `sign`, `post`) also risked cross-class remaps. Invariant enforced: one prompt -> exactly one `category_id`.
3. **"Statistically significant" (S3):** paired image-level bootstrap on CAL (1,000 resamples, SEED 42); a gain is significant when the 95% CI of the delta excludes 0. Thresholds are tuned per arm on CAL itself (optimistic); the held-out VAL comparison stays in T4 Section 3.
4. **Size buckets:** rasterized GT mask `area_ratio` (same as EXP-01 `recall_by_bucket`): `small < 0.0035 <= medium < 0.035 <= large`.
5. **Selective tiling subset:** categories whose CAL GT median `area_ratio < 0.035` (a training-side hyperparameter like the CAL threshold; never per-image; categories absent from CAL are included). The tile pass caches all 120 categories; the subset is applied offline and frozen to `artifacts/cache_exp02/tiling_categories.json`.
6. **"25% overlap":** overlap = 25% of the tile side, so `tile = dim / 1.75`. Any object narrower than the overlap band (~14% of W/H) lies fully inside some tile without touching its inner edge, which makes the inner-edge truncation filter safe.
7. **Edit surfaces beyond the T3 list (same pattern as T2):** `src/config_and_seeds.py` (aliases), `scripts/run_exp02_inference.py`, `scripts/build_notebook_02.py`, `notebooks/analysis_02/*.md`, `tests/test_perception_exp02.py`, `notebooks/README.md`.
8. **Backbone (resolved by human):** OWLv2 Base (controlled link to EXP-01) and OWLv2 Large (cached for T4), both run on a Kaggle/Colab T4 GPU.
9. **Cache layout fix (non-blocking):** the GPU caches arrived nested as `artifacts/cache_exp02/artifacts/cache_exp02/{base,large}` (archive path); flattened to `artifacts/cache_exp02/{base,large}` before use.
10. **Per-view thresholds (added during analysis):** a single threshold shared by the global and tile views raised the optimum 0.175 -> 0.225 and cost -8.1 pp large-object recall (significant). `merge_views` now takes `tile_min_score` and `tile_max_area_ratio` (defaults keep the old behaviour; unit-tested). `lvis_evaluator.box_match_details` (per-image TP/FP/FN + per-GT match flags, summing exactly to `evaluate_federated`) was added for the paired bootstrap; `_greedy_match` now delegates to the shared `_greedy_assign`.

### Results (T3 / S3) — CAL 80 images, box-only federated, 95% paired bootstrap (1,000 resamples, SEED 42)
- **Regression:** Base `single_raw` from the 183-query GPU pass reproduces the EXP-01 CPU cache exactly at metric level (P/R/F1 0.6883/0.6296/0.6576; 80/80 identical detection counts; max |score diff| 7e-4).
- **EXP-02A (global view; all arms optimal at threshold 0.175):** chunking +0.65 pp recall (significant), F1 +0.002 (n.s.); aliases -1.2 to -1.8 pp precision (significant), recall n.s.; sibling rule 0 suppressions at the operating point (per-block argmax already disambiguates). **OWLv2 Large < Base:** F1 -0.032, CI [-0.056, -0.008]. Selected: Base `chunked_raw`.
- **EXP-02B (selective tiling: global 0.175 + tiles 0.25, inner-edge filter 3 px, size gate area_ratio < 0.035, 106 tiled categories):** P 0.660 / R 0.674 / F1 0.6669; recall small 0.585 / medium 0.759 / large 0.800. Vs full image: small +6.4 pp [+4.6, +8.2], medium +1.4 pp, large unchanged, precision -2.5 pp, F1 +0.0073 (n.s.). Naive tiling (no filter) drops F1 to 0.607; the filter removes 27.8% of tile detections. The gain replicates in all 8 backbone x prompt arms (+2.8 to +6.7 pp small recall).
- **S3 vs EXP-01:** small-object recall 0.516 -> 0.585 (+6.9 pp, CI [+5.1, +8.7]) **met**; Box F1 +0.0093 (CI [-0.0024, +0.0210], p = 0.06) **positive but not significant** -> S3 partially met; thresholds are in-sample (CAL), VAL confirmation in T4.
- **Cost (T4, FP32, 5 views):** Base 2.65 s/img, 780 MB peak; Large 11.4 s/img, 2.0 GB peak. Chunking/aliases are free with OWLv2; tiling costs 5x.
- Notebook `notebooks/02_Perception_Ensemble_Tiling_EXP02.ipynb` (Sections 1-2; 36 cells, executed from cache on CPU, 0 errors), built by `scripts/build_notebook_02.py` with analyses in `notebooks/analysis_02/`.

### Notes for T4
- Reuse `artifacts/cache_exp02/{base,large}/detections_{variant}.json.gz` (180 images, tile views unfiltered), `tiling_categories.json` and `exp02ab_summary.json`; no new OWLv2 inference is needed for VAL.
- **Do not assume OWLv2 Large is the stronger family member:** it is significantly worse than Base here and 4.3x slower. Decide on CAL/VAL.
- Precision is now the bottleneck: calibrate scores per detector **and per view** (global vs tile) and let WBF/cross-view consensus promote corroborated boxes; this is the most likely route to a significant F1 gain.
- For MM-GDINO the 5 x 24 blocks are mandatory (256-token text limit); re-test aliases there (BERT encoder). Candidate sibling pair: `computer monitor` / `television set`.

---

## 6. T4 Build Log (Claude Code — 2026-10-29, branch `feat/t3-perception-chunk-tiling`)

### Spec-Challenge Notes (T4 / S4)
1. **OWLv2 member chosen on CAL, not by name (non-blocking):** S4 names `OWLv2 Large`, but T3 measured Large
   significantly below Base (F1 -0.032, CI [-0.056, -0.008]). The ensemble evaluates {Base, Large, Base+Large} x
   MM-GDINO and selects by max CAL Box F1; VAL is only reported.
2. **MM-GDINO checkpoint (non-blocking):** `openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg` (Swin-L,
   zero-shot). `*_large_all` is excluded: its pre-training includes COCO/LVIS-family data, weakening the zero-shot
   claim. `IDEA-Research/grounding-dino-base` stays selectable in the runner as a fallback.
3. **YOLO-Worldv2-X skipped (non-blocking):** needs `ultralytics` (new AGPL dependency, Risk item 5) with no HF path;
   S4 does not require it and the deadline is 2026-10-31.
4. **S4 bar is weak (non-blocking):** EXP-01 already reaches VAL Box F1 0.639 > 0.60. The reported claim is EXP-02C vs
   EXP-01 and vs EXP-02B on VAL as paired 95% bootstrap CIs.
5. **Segmentation unchanged (non-blocking):** SAM2.1 Base+ with `multimask_output=False` as in EXP-01, so the Mask F1
   change is attributable to the boxes; multimask / Large belongs to T5.
6. **Definitions:** "same detector, same view" = identical `Detection.source` (`owlv2:global`, `mmgdino:tile2`);
   "high-confidence" = calibrated score >= 0.50 (canonical); fused score = sum over detectors of their best member's
   calibrated score / number of ensemble detectors (single-family boxes are halved, consensus is promoted).
7. **Calibration per (detector, view kind in {global, tile})**, since T3 showed the views have different scales.
8. **Sensitivity groups from CAL only, frozen to JSON:** `micro` = CAL GT median `area_ratio < 0.0035`; `fp_prone` =
   >= 3 verified neg-FPs on CAL with neg share >= 50% at the canonical threshold (priority over micro); `standard` =
   the rest, including categories absent from CAL.
9. **Edit surfaces beyond the T4 list (same pattern as T3):** new `src/detection_fusion.py` (keeps
   `perception_detectors.py` < 400 lines), `tests/test_detection_fusion.py`, `tests/test_perception_exp02.py`,
   `scripts/run_exp02_inference.py`, new `scripts/run_exp02c_segmentation.py`, `scripts/build_notebook_02.py`,
   `notebooks/analysis_02/*.md`, `notebooks/README.md`, new `notebooks/kaggle/T4_EXP02C_runner.ipynb`, `.gitignore`.
10. **Lineage:** `Detection` gains `raw_score: float | None = None` (last field); existing caches load unchanged.
11. **Compute:** MM-GDINO inference and SAM2 run on a Kaggle/Colab T4 (the workstation has no CUDA); the GDINO forward
    path has no local automated test (Risk item 6), mitigated by a `--limit 2` smoke run before the full pass.

### Results (T4 / S4) — Kaggle Tesla T4; CAL 80 in-sample, VAL 40 held out; 95% paired bootstrap (1,000 resamples, SEED 42)
- **Second family:** MM-GDINO Large alone ~ OWLv2 Base (CAL tiled F1 0.6645 vs 0.6669, n.s.), but complementary. Oracle union small recall is 0.695 vs 0.585. MM-GDINO is much stronger on large objects (R_large 0.881 vs 0.800).
- **Calibration:** the raw F1-optimal thresholds coincide (tau = 0.175 for both families, global view). What differs is the distribution shape: 22% vs 7% of CAL global detections score >= 0.3. Under NMS, calibration is neutral (C4 - C3 F1 +0.001, n.s.); it is needed for WBF's score sum.
- **CAL ablation:**
  - Ensemble vs OWLv2 alone: C6 - C1 F1 **+0.058 [+0.034, +0.079]**.
  - WBF vs NMS: C6 - C4 F1 +0.016 (p = 0.051); R +4.8 pp and R_small +5.8 pp significant.
  - Member set: Base+GDINO 0.7236 > Large+GDINO 0.7081; Base+Large+GDINO 0.7234 adds nothing. Selected: **OWLv2 Base + MM-GDINO**.
- **Non-self-collision rule: inert.** 0 of 64,944 same-view same-class pairs reach IoU >= 0.55, because the per-view NMS at 0.5 already prevents collapse. The rule fires only via fused-box drift (3 boxes at the operating point on DEV) and changes no match. Kept as a tested invariant; no gain attributed.
- **Groups:**
  - micro 0.425 / standard 0.450. `fp_prone` is empty: verified-negative FPs are ~1% of box FPs (DEV: 6 for EXP-02B, 3 for EXP-02C).
  - C7 - C6 F1 +0.004 (n.s.), R_small +2.4 pp (significant).
- **S4 (VAL, held out): MET.** Box F1 EXP-01 0.6389 / EXP-02A 0.6364 / EXP-02B 0.6416 / **EXP-02C 0.6822**.
  - vs EXP-01: +0.0434 [+0.013, +0.080]. vs EXP-02B: +0.0406 [+0.009, +0.079]; precision and recall rise together.
  - CAL -> VAL optimism is 0.045, the largest of the four arms. The honest effect size is +0.04, not +0.07. The master-plan target of ">= +8 points" is not reached (+4.3 on VAL).
- **Masks (SAM2.1 Base+, single mask, same as EXP-01):** VAL Mask F1 0.6198 -> 0.6655 (+0.0457 [+0.018, +0.083]); `0.40 Box + 0.35 Mask` +0.0333 [+0.013, +0.059]. 0 mask fallbacks over 6,850 objects; 92% of submitted objects are cross-family consensus boxes.
- **Submission #2:** `submissions/submission_exp02_perception.csv`, 80 rows. Suite **8/8 PASS** on the GPU (live blind inference + VRAM hooks); 0 FAIL locally without hooks.
- **Cost (T4, FP32):** OWLv2 Base 2.6 s + MM-GDINO Large 62.7 s (25 forwards; n = 150 traced, one resumed run lost the trace of 30 images) + SAM2 0.54 s + build 0.33 s ≈ 66 s/img; peak VRAM 8.3 GB.
- **Frozen config:** `artifacts/cache_exp02/exp02c/config.json` (sha `912f09d78ec5dd6e`). The notebook's analysis Markdown was inserted with `build_notebook_02.py --refresh-markdown`; code cells and GPU outputs are untouched.

### Notes for T5
- Freeze the EXP-02C boxes by config sha `912f09d78ec5dd6e`. The fused detections are rebuilt blind by `scripts/run_exp02c_segmentation.fused_detections`. `exp02c/scenes.jsonl` (54 MB) is git-ignored: unzip it from the Kaggle outputs or regenerate it on a GPU.
- Masks are now the weaker term (VAL Mask F1 0.666 vs Box F1 0.682) with 0 fallbacks: the gap is mask quality, which is exactly EXP-03's variable (SAM2.1 Large, `multimask_output=True`, box-aware selection).
- MM-GDINO dominates latency. If time matters: FP16, or batch the 5 blocks of a view into one forward.
- Open: the MM-GDINO alias variant was not run (`RUN_ALIAS = False`).
- Operational gotcha: on this workstation `cat` and `ls` are aliased (`bat`, `eza`), so use `command cat` / Python for heredoc writes. Kaggle zips unpack into a `t4_outputs/` folder, not the repo root.
