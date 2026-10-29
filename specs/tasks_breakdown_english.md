# Tasks: TPI Visual Scene Understanding (`tpi-clean-slate`)

> **Handoff Target:** Claude Code (Build Agent)  
> **Spec Reference:** `specs/tpi-clean-slate/spec.md`  
> **Master Plan Reference:** `Plan Maestro Integral desde Cero — TPI Comprensión Visual de Escenas (Primavera 2026).md`

---

## Task Slices (`T1` .. `T8`)

### Task T1: Contractual Foundations, Stratified Partition, and Federated LVIS Evaluator
- **Status:** Done (2026-10-28, branch `feat/t1-contract-foundations`; 32 unit tests OK; S1 oracle PASS). Commit: `481400a` (feat), docs commit follows
- **Maps to:** `S1`
- **Risk Tier:** High (Normative foundation for the 15% JSON Consistency and local LVIS metric)
- **Claude Code Effort Level:** `medium` (`/effort medium`)
- **Scope (`<= 400` lines per module):**
  - `src/config_and_seeds.py`
  - `src/trace_logger.py`
  - `src/scene_builder.py`
  - `src/lvis_evaluator.py`
  - `artifacts/split_cal_val.json`
- **Verification Command:**
  - `python3 -m unittest discover -s tests -p "test_contracts.py"` (or direct execution of `sample_submission.csv` validation and GT oracle `Box F1 == 1.0, Mask F1 == 1.0`).

---

### Task T2: Deterministic QA Engine + Notebook `01_EDA_and_Baseline_EXP01.ipynb` + Submission #1
- **Status:** Done (2026-10-29, branch `feat/t2-qa-engine-baseline`; 60 unit tests OK; DEV Box F1 0.652 / Mask F1 0.634 at CAL threshold 0.175; suite 7 PASS / Test 8 SKIPPED on CPU / 0 FAIL). Commits: `ca8b235` (QA engine), `6f25a2f` (detector/segmenter/pipeline), `2519b13` (device auto-select), `9b77f8f` (notebook + submission), docs commit follows
- **Maps to:** `S2`
- **Dependencies:** `T1`
- **Risk Tier:** High (First control submission to Kaggle)
- **Claude Code Effort Level:** `high` (`/effort high`)
- **Scope:**
  - `src/qa_engine.py` (deterministic mode)
  - `src/perception_detectors.py` (baseline)
  - `src/segmentation_sam2.py` (baseline)
  - `notebooks/01_EDA_and_Baseline_EXP01.ipynb`
  - `submissions/submission_exp01_baseline.csv`
- **Verification Command:**
  - `python3 -c "from src.lvis_evaluator import run_full_verification_suite; run_full_verification_suite('submissions/submission_exp01_baseline.csv')"`

---

### Task T3: Perception `EXP-02A/B` — Chunked Prompting, LVIS Aliases, and 2×2 Tiling
- **Status:** Implemented (2026-10-29, branch `feat/t3-perception-chunk-tiling`; 86 unit tests OK; OWLv2 Base + Large caches from a Kaggle T4). **S3 partially met — pending human acceptance:** CAL small-object Box Recall +6.9 pp vs EXP-01 (95% CI [+5.1, +8.7], significant); CAL Box F1 0.6576 -> 0.6669 (+0.0093, CI [-0.0024, +0.0210], not significant at 95%). Commits: `e241dca` (spec-challenge notes), `fb1b0d5` (chunked prompting + aliases), `1303309` (2x2 tiling), `b46ff23` (GPU runner), `e7efe7c` (per-image matching + per-view tile policy), `2d82181` (notebook + caches), docs commit follows
- **Maps to:** `S3`
- **Dependencies:** `T2`
- **Risk Tier:** Medium
- **Claude Code Effort Level:** `high` (`/effort high`)
- **Scope:**
  - `src/perception_detectors.py` (`Chunked Prompting`, `LVIS_PROMPT_ALIASES`, `Tiling 2x2`)
  - `notebooks/02_Perception_Ensemble_Tiling_EXP02.ipynb` (Sections 1 and 2)
- **Verification Command:**
  - Verification in cells of `02_Perception_Ensemble_Tiling_EXP02.ipynb` confirming improvement of `Box Recall` on `area_ratio < 0.0035` over `CAL`.

---

### Task T4: Perception `EXP-02C` — Calibrated Multi-Family Ensemble, Anti-Collision WBF, and Submission #2
- **Maps to:** `S4`
- **Dependencies:** `T3`
- **Risk Tier:** High (Core of the 40% `Box F1`)
- **Claude Code Effort Level:** `max` (`/effort max` / `ultrathink`)
- **Scope:**
  - `src/perception_detectors.py` (Quantile calibration on `CAL` + WBF with non-self-collision constraint)
  - `notebooks/02_Perception_Ensemble_Tiling_EXP02.ipynb` (Sections 3, 4, and 5)
  - `submissions/submission_exp02_perception.csv`
- **Verification Command:**
  - `python3 -c "from src.lvis_evaluator import run_full_verification_suite; run_full_verification_suite('submissions/submission_exp02_perception.csv')"`

---

### Task T5: Segmentation `EXP-03` — `SAM2.1 Large` with Box-Aware Multimask Selection
- **Maps to:** `S5`
- **Dependencies:** `T4`
- **Risk Tier:** Medium (Core of the 35% `Mask F1`)
- **Claude Code Effort Level:** `high` (`/effort high`)
- **Scope:**
  - `src/segmentation_sam2.py` (`multimask_output=True`, `Box-Mask IoU Consistency`, guarantee $\sum M \ge 1$)
  - `notebooks/03_SAM2_Multimask_and_SLM_QA_EXP03_EXP04.ipynb` (Part 1)
- **Verification Command:**
  - Verification on `CAL` and `VAL` of `Mask F1` increase and `0` empty masks.

---

### Task T6: Language `EXP-04` — SLM Interpreter (`Qwen2.5-3B-Instruct`), Guardrails, and Submission #3
- **Maps to:** `S6`
- **Dependencies:** `T5`
- **Risk Tier:** High (Mandatory requirement for Direct Pass / Promoción + 10% QA)
- **Claude Code Effort Level:** `high` (`/effort high`)
- **Scope:**
  - `src/qa_engine.py` (`SLMQuestionInterpreter`, `SemanticCompletenessGuard`, factual description verifier)
  - `notebooks/03_SAM2_Multimask_and_SLM_QA_EXP03_EXP04.ipynb` (Part 2)
  - `submissions/submission_exp04_integrated.csv`
- **Verification Command:**
  - `python3 -c "from src.lvis_evaluator import run_full_verification_suite; run_full_verification_suite('submissions/submission_exp04_integrated.csv')"`

---

### Task T7: Outstanding Extension `EXP-05` — Consensus Rescue and Collective RLE Suppression
- **Maps to:** `S7`
- **Dependencies:** `T6`
- **Risk Tier:** High (Differential requirement for Outstanding / Sobresaliente)
- **Claude Code Effort Level:** `max` (`/effort max` / `ultrathink`)
- **Scope:**
  - `src/perception_detectors.py` (1-to-1 inter-family rescue)
  - `src/segmentation_sam2.py` (Collective box suppression covered $\ge 75\%$ by mask union)
  - `notebooks/04_Extension_EXP05.ipynb`
  - `submissions/submission_exp05_sobresaliente.csv`
- **Verification Command:**
  - `python3 -c "from src.lvis_evaluator import run_full_verification_suite; run_full_verification_suite('submissions/submission_exp05_sobresaliente.csv')"`

---

### Task T8: Consolidation of Self-Contained Final Notebook `TPI_IA_2026P_2.ipynb` and `REPORTE_TECNICO.md`
- **Maps to:** `S8`
- **Dependencies:** `T7`
- **Risk Tier:** High (Final official deliverables)
- **Claude Code Effort Level:** `high` (`/effort high`)
- **Scope:**
  - `TPI_IA_2026P_2.ipynb`
  - `REPORTE_TECNICO.md`
  - `submission.csv`
- **Verification Command:**
  - `python3 -c "from src.lvis_evaluator import run_full_verification_suite; run_full_verification_suite('submission.csv')"`