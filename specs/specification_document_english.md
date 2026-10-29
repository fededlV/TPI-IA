# Spec: TPI Visual Scene Understanding — Clean-Slate Outstanding & Top-1 Kaggle

- **Status:** Approved
- **Approved By:** Fede (Group 2)
- **Approval Date:** 2026-10-28
- **Change Slug:** `tpi-clean-slate`

---

## 1. Objective

Build from scratch (*clean-slate*) a modular, traceable, and reproducible **Visual Scene Understanding** system (`Perception -> Segmentation -> Representation -> Reasoning -> Interaction`) for the Spring 2026 TPI (UTN — Artificial Intelligence Course, **Group 2**), implemented in Jupyter Notebooks and designed to:
1. Fulfill all validity, **Pass (Aprobado)**, **Direct Pass (Promoción)**, and **Outstanding (Sobresaliente)** conditions of the Official Guidelines.
2. Maximize the `PipelineScore` on Kaggle (`0.40 * Box F1 + 0.35 * Mask F1 + 0.15 * JSON Consistency + 0.10 * QA Benchmark`) with 4 iterative control submissions before the **2026-10-31** deadline.

---

## 2. Normative Contracts & Invariants

1. **Zero Ground-Truth Leakage & Zero End-to-End VLM:**
   - It is strictly forbidden to consult `annotations.json` during inference, filter classes per image using GT data, or send the raw image to a VLM/LLM.
   - `interpret_question_with_llm(question_text, categories_120)` receives exclusively the question text and the list of 120 categories.
2. **Exact Schema Compliance (`scene_schema.json` & `sample_submission.csv`):**
   - Required root keys in `SceneJSON`: `["image", "objects", "relations"]` (plus `"questions"` and `"questions_llm"` in the 20 development rows of `questions_dev.jsonl`).
   - Required keys per object in `"objects"` (`maxItems: 300`): `["id", "category_id", "confidence", "bbox", "mask", "centroid", "area_ratio", "position", "size"]`.
   - `bbox` in `SceneJSON` expressed as `[x1, y1, x2, y2]` (`xyxy`), whereas `annotations.json` uses `[x, y, w, h]` (`xywh`).
   - `mask` encoded as uncompressed RLE in `column-major` order (`size: [H, W]`, `counts: [...]` starting with the zero count) and `0 < area_ratio <= 1` (`exclusiveMinimum: 0`, guaranteeing $\sum M \ge 1$).
   - `relations` (`maxItems: 20000`) with symmetric predicates `["left_of", "right_of", "above", "below", "overlaps"]` by pure centroid comparison and intersection.
   - In `questions` and `questions_llm` (60 prompts across 20 dev images: 20 `description` + 40 closed questions), `normalized_answer` for `type == "count"` must be an `int` (`0, 1, 2...`), except when `"unsupported"`.

---

## 3. Acceptance Criteria (Given / When / Then — TDD Ready)

- **S1 (Contract Audit and Stratified Partition):**
  - *Given* the normative files `scene_schema.json`, `sample_submission.csv`, `categories.csv`, `questions_dev.jsonl`, and `development/annotations.json`,
  - *When* the `scene_builder` constructor and the `lvis_evaluator` are executed on the ground truth of the 20 images in `questions_dev.jsonl`,
  - *Then* all 80 rows validate with `0` errors against `scene_schema.json`, the LVIS evaluator reports `Box F1 = 1.0` and `Mask F1 = 1.0` against the GT itself, and `artifacts/split_cal_val.json` is generated (`CAL`: 80 images with 10 QA; `VAL`: 40 images with 10 QA, `SEED = 42`).

- **S2 (Actionable EDA and End-to-End Baseline — Submission #1):**
  - *Given* the `CAL`/`VAL` partition and the deterministic `qa_engine`,
  - *When* `notebooks/01_EDA_and_Baseline_EXP01.ipynb` is executed,
  - *Then* the 4 visual EDA analyses documented in Markdown are generated, the baseline (`EXP-01`) is measured with `Box F1 >= 0.45` on development, and `submissions/submission_exp01_baseline.csv` (80 rows) is produced passing all 8 tests of `run_full_verification_suite()`.

- **S3 (Perception with Chunked Prompting, LVIS Aliases, and 2×2 Tiling — EXP-02A/B):**
  - *Given* the 120-category vocabulary where 71.06% of GT instances have `area_ratio < 0.01`,
  - *When* `EXP-02A` (5 semantic blocks + `LVIS_PROMPT_ALIASES`) and `EXP-02B` (global pass + 2×2 tiling with 25% overlap) are executed on `CAL`,
  - *Then* the `Box Recall` on small objects (`area_ratio < 0.0035`) and the global `Box F1` statistically significantly outperform Baseline `EXP-01`.

- **S4 (Calibrated Multi-Family Ensemble and Anti-Collision WBF — EXP-02C / Submission #2):**
  - *Given* the detections from `OWLv2 Large` and `MM-Grounding-DINO Large` with heterogeneous confidence scales,
  - *When* monotonic calibration fitted on `CAL` and WBF fusion with intra-detector non-self-collision constraints are applied,
  - *Then* `Box F1` on `VAL` exceeds `0.60+` (well above the `0.50` required for Direct Pass / Promoción) and a valid `submissions/submission_exp02_perception.csv` is generated for Submission #2.

- **S5 (SAM2.1 Large Box-Aware Multimask Segmentation — EXP-03):**
  - *Given* the frozen boxes from the best detector of `EXP-02C`,
  - *When* `SAM2.1 Large` is executed with `multimask_output=True` and selection via `Score_final(k) = alpha * s_k + (1 - alpha) * IoU_box(k)` guaranteeing $\sum M \ge 1$,
  - *Then* `Mask F1` improves over `multimask_output=False` and no mask across the 80 rows violates `area_ratio > 0` or the RLE sum `H * W`.

- **S6 (Image-Blind SLM Interpreter, Guardrails, and Factual Verifier — EXP-04 / Submission #3):**
  - *Given* `Qwen/Qwen2.5-3B-Instruct` (`do_sample=False`) and the 60 prompts in `questions_dev.jsonl` plus 12 out-of-scope/ambiguous queries,
  - *When* `resolve_llm_qa` is executed verifying via `assert` that the SLM does not receive `SceneJSON` or structured fields when interpreting the question,
  - *Then* it achieves $\ge 95\%$ agreement with the deterministic engine on the 40 closed questions, classifies 100% of out-of-scope queries as `unsupported`, generates hallucination-free descriptions, and produces `submissions/submission_exp04_integrated.csv`.

- **S7 (Outstanding-Level Extension and Critical Analysis — EXP-05):**
  - *Given* the `Mask-Guided Collective Suppression & Inter-Family Consensus Rescue` module,
  - *When* its two components are evaluated separately and combined in `notebooks/04_Extension_EXP05.ipynb`,
  - *Then* the improvement in `Box F1`, `Mask F1`, and `Count Accuracy` is quantitatively documented, along with 2 visual success cases, 1 limitation/failure case, the cost in `ms/img` and VRAM, and the error propagation traceability across 3 scenes.

- **S8 (Consolidation of Self-Contained Final Notebook and Technical Report — Submission #4):**
  - *Given* all validated modules and experiments,
  - *When* `TPI_IA_2026P_2.ipynb` is executed end-to-end and `REPORTE_TECNICO.md` is generated,
  - *Then* all cells are saved with their outputs without errors, `submission.csv` passes all 8 tests of `run_full_verification_suite()`, and the values in `REPORTE_TECNICO.md` match the notebook exactly.

---

## 4. Assumptions

1. The official dataset is available locally at `/home/fede/Ingenieria_en_Sistemas/Inteligencia Artificial/Tp/dataset/` and mounted automatically on Kaggle when opening the competition notebook.
2. A CUDA-capable GPU is available (local or Kaggle T4 with 15–16 GB VRAM) to run `OWLv2 Large`, `MM-Grounding-DINO Large`, `SAM2.1 Large`, and `Qwen2.5-3B-Instruct` in FP16 while freeing VRAM between stages (`torch.cuda.empty_cache()`).

---

## 5. Boundaries / Out of Scope

- **Always do:** Validate all 80 rows against `scene_schema.json` before each submission; fix random seeds (`SEED = 42`); pair every code cell with a technical analysis Markdown cell; free VRAM between stages.
- **Ask first:** Changing the SLM model if `Qwen2.5-3B-Instruct` is not cached; modifying the official filename `TPI_IA_2026P_2.ipynb`.
- **Never do:** Use annotations from `annotations.json` during inference; pass images to a VLM; hardcode answers by `image_id` or `question_id`; emit empty masks (`area_ratio == 0`) or strings in `normalized_answer` for `type == "count"`.

---

## 6. Open Questions

- *(Resolved)* Group number confirmed: **Group 2** (`TPI_IA_2026P_2.ipynb`).