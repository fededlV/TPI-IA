# **Comprehensive Clean-Slate Master Plan for AI Agents**

# **Integrative Practical Assignment (TPI) — Visual Scene Understanding (Spring 2026)**

**Objective:** Achieve an **Outstanding (Sobresaliente)** grade in the academic evaluation and compete for **First Place on the Kaggle Leaderboard** (surpassing the `0.82+` `PipelineScore` barrier) through clean-slate engineering that is modular, traceable, implemented in reproducible **Jupyter Notebooks**, and validated with iterative control submissions (`submission.csv`) before the **October 31, 2026** deadline (and final academic submission by **November 1, 2026 at 23:59 hs**).

---

# **1. Deep Analysis of the Official Guidelines and Normative Dataset Audit**

## **1.1 Conceptual Progression and Course Philosophy**

The system must transform a raw image into a structured intermediate representation (`SceneJSON`), factually describe its content, and answer closed questions about it strictly following the sequential chain:

$$\text{Perception} \longrightarrow \text{Segmentation} \longrightarrow \text{Representation} \longrightarrow \text{Reasoning} \longrightarrow \text{Interaction}$$

The course instructors evaluate six fundamental pillars that every AI Agent must respect when writing code and Markdown cells in the notebooks and when designing experiments:

1. **Design of a modular and traceable pipeline:** Every stage must have inputs, outputs, execution times (`duration_ms`), memory usage (`vram_peak_mb`), and auditable states logged per image.
2. **Hypothesis formulation and controlled experiments:** Every improvement must be compared against a *baseline* by modifying one variable or controlled component at a time. Attempts that did not improve the system (negative evidence) must also be documented when relevant.
3. **Stage-wise evaluation with appropriate metrics:** Independently measure detection (`Box Precision`, `Box Recall`, `Box F1`, `LVIS Coverage`), segmentation (`Mask F1`, `Mask IoU`), structured representation consistency (`JSON Schema` + geometric invariants), and question answering (`Accuracy` / deterministic vs. LLM equivalence).
4. **Error analysis and propagation:** Identify how a failure originating in detection or in the SAM2 mask alters derived geometric attributes (`centroid`, `area_ratio`, `position`, `size`, `relations`) and final answers.
5. **Strict epistemological distinction:** Explicitly separate in code, traces, and visualizations the **observed** information (boxes, scores, masks), the **derived** information (`area_ratio`, `centroid`, `position` 3×3, `size`, `dominant_color`, `relations`), and the **unknown or out-of-scope** information (`unsupported` / `insufficient_information`).
6. **Technical justification and narrative (*Technical Storytelling*):** Every model, threshold, resolution, and fusion rule must be backed by Exploratory Data Analysis (EDA) data and quantitative ablations. Every code cell in the notebooks must be accompanied by its analysis and interpretation in Markdown.

## **1.2 Strict Prohibitions (Unbreakable Validity Rules)**

Violating any of these rules invalidates the work and results in direct failure:

* **PROHIBITION 1 — Zero End-to-End Vision-Language Models (VLM):** It is strictly forbidden to send the original image to a VLM or to the question-answering/description LLM. The sole source of visual information for answering questions or writing descriptions is the predicted `SceneJSON`.
* **PROHIBITION 2 — Absolute Isolation of Ground Truth During Inference:** It is forbidden to consult `annotations.json` from the 120 development images to assist inference (forbidden to filter categories present per image, fill in missing objects, correct boxes/masks, or assist answers). The main inference function must be able to receive an arbitrary external image without annotations (a condition verifiable by the instructors during the oral defense).
* **PROHIBITION 3 — Integrity of the Test Set (60 images):** It is forbidden to identify the original images in LVIS/COCO, search for or download their external labels, manually label them, or apply case-by-case manual corrections (*hardcoding* by `image_id` or `question_id`).
* **PROHIBITION 4 — Isolation of the LLM Interpreter (`questions_llm`):** In LLM mode, the model receives exclusively the raw question text (`question`) and the allowed vocabulary/schema. It cannot access the structured question fields reserved for deterministic mode (`class_en`, `class_es`, `type`, `answer_format`, `allowed_answers`), nor replace the Python deterministic engine, nor compute counts on its own, nor invent data.
* **PROHIBITION 5 — Self-Contained Notebook Without Hidden Dependencies:** The final notebook (`TPI_IA_2026P_2.ipynb`) must execute entirely on Kaggle using only the mounted official dataset and public resources downloaded by the notebook itself, with fixed seeds and all outputs saved.
* **PROHIBITION 6 — Zero Artificial Manipulation of Metrics:** Any artificial manipulation of results, metrics, or outputs in the notebook cells or in the report is grounds for failure.

## **1.3 LVIS Federated Evaluation Policy (Key to Optimizing Precision and Recall)**

The dataset consists of **120 development images** (`tpi_0001.jpg` to `tpi_0120.jpg`, with **1,790 annotated instances** in `annotations.json`) and **60 test images** (`test_0001.jpg` to `test_0060.jpg`) over an evaluable vocabulary of **120 categories** from LVIS/COCO (`categories.csv`, IDs `1` to `120`). Because LVIS uses federated annotation, the absence of an annotation does not mean the object is not in the image. The evaluator classifies each category $c$ in an image $I$ into four states:

1. **Positive category ($c \in \text{GT}(I)$):** Has one or more annotated instances in `annotations.json`. Omitting an annotated instance generates a **False Negative (FN)** and directly penalizes Recall.
2. **Category verified as absent ($c \in \text{neg\_category\_ids}(I)$):** Verified not to exist in the image. Predicting an instance of this category generates a **False Positive (FP)** and penalizes Precision.
3. **Non-exhaustive positive category ($c \in \text{not\_exhaustive\_category\_ids}(I)$):** There are annotated instances, but more unannotated instances may exist. Predictions matching ($\text{IoU} \ge 0.5$) annotated instances count as **True Positives (TP)**; additional predictions that do not match annotated instances **are ignored** (do not add FP).
4. **Unverified category ($c \notin \text{GT}(I) \cup \text{neg\_category\_ids}(I)$):** Neither has annotated instances nor appears in `neg_category_ids`. Its predictions **are ignored** when computing false positives (although **evaluation coverage**—i.e., the fraction of predictions falling into verified evaluable categories—must be reported).

## **1.4 Breakdown of the Leaderboard Equation (`PipelineScore`)**

The automatic score on Kaggle is governed by the weighted formula:

$$\text{PipelineScore} = 0.40 \times \text{Box } F_1 + 0.35 \times \text{Mask } F_1 + 0.15 \times \text{JSON Consistency} + 0.10 \times \text{QA Benchmark}$$

* **40% `Box F1` and 35% `Mask F1` (75% of the total):** Evaluated at $\text{IoU} \ge 0.5$ under the LVIS federated policy on the test set (60 images). Pass (Aprobado) requires $\text{Box } F_1 \ge 0.40$ on development; Direct Pass (Promoción)/Outstanding (Sobresaliente) requires $\text{Box } F_1 \ge 0.50$.
* **15% Structured Representation Consistency:** Evaluates strict compliance with `scene_schema.json` and the mathematical accuracy of derived properties (`centroid`, `area_ratio`, `position` in a 3×3 grid, `size` in `small/medium/large`, `relations`, and RLE masks in `column-major` order).
* **10% Question Answering and Description Benchmark:** Computed over the 20 development images that have associated prompts in `questions_dev.jsonl` (`tpi_0001` to `tpi_0012`, `tpi_0014`, `tpi_0015`, `tpi_0019`, `tpi_0020`, `tpi_0022`, `tpi_0023`, `tpi_0024`, `tpi_0025`: 20 descriptions + 40 closed questions), averaging the deterministic mode (`questions`) and the LLM mode (`questions_llm`).

---

# **2. Critical Corrections and Reverse Engineering of the Official Contract (`scene_schema.json`, `sample_submission.csv`, and Real Dataset)**

> [!CAUTION]
> **Normative Audit Findings (Mandatory Corrections Over Previous Plan Versions):**
> By contrasting the plan with the official files in `/home/fede/Ingenieria_en_Sistemas/Inteligencia Artificial/Tp/dataset/` (`scene_schema.json`, `sample_submission.csv`, `categories.csv`, `questions_dev.jsonl`, and `annotations.json`), **7 strict technical contracts** were identified that prevent silent losses of up to 25 points on the Leaderboard:

1. **Exact Field Names in `scene_schema.json` (Correction of `area_ratio` and `relations`):**
   * At the root of `SceneJSON`, the required keys are exactly `["image", "objects", "relations"]` (and in the 20 development images, `"questions"` and `"questions_llm"` are added). **Never** name the root field `spatial_relations`.
   * In each element of `"objects"`, the 9 required keys are exactly:
     `["id", "category_id", "confidence", "bbox", "mask", "centroid", "area_ratio", "position", "size"]`.
     **Never** use `relative_area` instead of **`area_ratio`**. If extended attributes are added (such as `dominant_color` or `category_name`), they are added as additional keys (`additionalProperties: true`), keeping the 9 mandatory ones intact.
2. **`"exclusiveMinimum": 0` Constraint on `area_ratio` (Protection Against Empty Masks):**
   * In `scene_schema.json` (line 134), `area_ratio` requires `"exclusiveMinimum": 0` ($\text{area\_ratio} > 0$). If morphological post-processing empties a small object's mask (leaving 0 active pixels), `area_ratio` becomes `0.0` and **the entire row fails JSON Schema validation**.
   * **Shielding rule:** No object emitted in `SceneJSON["objects"]` may have an empty mask ($\sum \text{mask} \ge 1$). If connected-component cleaning or SAM2 produces an empty mask, automatically fall back to the original SAM2 mask or the rectangular `bbox` fill. The 1-pixel erosion used to compute `dominant_color` is applied **strictly on a temporary in-memory copy** (falling back to the un-eroded mask if the copy becomes empty).
3. **Box Format Difference (`xywh` in `annotations.json` vs. `xyxy` in `SceneJSON`):**
   * `development/annotations.json` stores ground truth boxes in COCO format **`[x, y, width, height]` (`xywh`)**.
   * `SceneJSON` (`submission.csv`) requires boxes in **`[x1, y1, x2, y2]` (`xyxy`)** format with $0 \le x_1 < x_2 \le W$ and $0 \le y_1 < y_2 \le H$.
   * Similarly, in `annotations.json` and `questions_dev.jsonl`, `image_id` is an integer (`1`..`120`), whereas in `submission.csv` and `SceneJSON["image"]["id"]` it is a **string without extension** (`"test_0001"`..`"test_0060"`, `"tpi_0001"`..`"tpi_0025"`).
   * In `SceneJSON["image"]`, dimensions are `"width": W, "height": H`, whereas in `mask["size"]` the order is **`[H, W]` (`[height, width]`)** and `mask["counts"]` is encoded in **column-major order (Fortran order, `mask.T.flatten()` or `np.asfortranarray(mask).ravel(order='F')`)** always starting with the count of `0` pixels (if the first pixel `(0,0)` is `1`, the first element of `counts` is `0`).
4. **Real Size Distribution in `annotations.json` (71.06% Are Micro-Objects) and `size` Thresholds:**
   * Across the 1,790 real instances in `annotations.json`, the percentiles of `area_ratio` are:
     * P10 = `0.00018`, P25 = `0.00080`, P33 = `0.00126`, **Median (P50) = `0.00289`**, P66 = `0.00766`, P75 = `0.01270`, P90 = `0.04683`.
     * **71.06% of objects occupy less than 1% of the image (`area_ratio < 0.01`).**
   * In the 5 `size` questions of `questions_dev.jsonl`, the queried objects have mask `area_ratio`:
     `necklace` (`0.00013`), `watch` (`0.00041`), `spectacles` (`0.00172`), `elephant` (`0.00560`, area $= 1,520\text{ px} > 32^2$), and `tablecloth` (`0.07540`, area $= 20,461\text{ px} > 96^2$).
   * **Canonical Mathematical Rule for `size` (Aligned with COCO $32^2 / 96^2$ relative to typical $640 \times 480 = 307,200\text{ px}$ resolution, where $32^2 / 307200 \approx 0.00333$ and $96^2 / 307200 \approx 0.0300$):**
     * `small`: $\text{area\_ratio} < 0.0035$
     * `medium`: $0.0035 \le \text{area\_ratio} < 0.0350$
     * `large`: $\text{area\_ratio} \ge 0.0350$
     *(With this physical COCO-relative convention, `necklace`, `watch`, and `spectacles` are `small`, `elephant` in `tpi_0025` is `medium`, and `tablecloth` in `tpi_0024` is `large`)*.
5. **Complete and Symmetric Definition of `SceneJSON["relations"]` (Without Cone Clipping):**
   * In `questions_dev.jsonl`, relation questions like `DEV-5-Q2` (`clock` at `(0.15, 0.04)` `above` `ring` at `(0.58, 0.64)`) or `DEV-15-Q1` (`bottle` at `(0.05, 0.73)` `left_of` `strap` at `(0.93, 0.28)`) evaluate pairs of objects that **do not overlap on the perpendicular axis**.
   * If relations are filtered by distance or angular cone, these pairs disappear from `SceneJSON["relations"]` and the questions fail.
   * Therefore (`scene_schema.json` allows up to `maxItems: 20000` in `relations`), for every ordered pair $(A, B)$ with $A \neq B$ (over all detected objects, or the top-$K=95$ highest-confidence objects if $N > 95$ to guarantee $\text{len}(\text{relations}) < 20,000$):
     * `left_of`: `cx_A < cx_B` (and symmetrically `right_of`: `cx_A > cx_B`)
     * `above`: `cy_A < cy_B` (and symmetrically `below`: `cy_A > cy_B`)
     * `overlaps`: $\text{IoU}_{\text{bbox}}(A, B) > 0$ or mask intersection $> 0$ (symmetric).
6. **Exact Contract of `questions` and `questions_llm` on `questions_dev.jsonl`:**
   * `questions_dev.jsonl` contains **60 entries** (3 for each of the 20 dev images: 1 of `type: "description"` and 2 closed questions).
   * **Both** arrays (`SceneJSON["questions"]` and `SceneJSON["questions_llm"]`) must include all **3 entries per image** (the 20 descriptions also go into `questions` generated via a pure deterministic template from `SceneJSON`, and into `questions_llm` verbalized by the SLM and validated against `SceneJSON`).
   * In `questions_dev.jsonl`, questions with `type == "relation"` provide `"class_en": [subject_class, object_class]`, but **do not include a separate `predicate` field**: the deterministic engine must extract the predicate (`above`, `below`, `left_of`, `right_of`, `overlaps`) from the canonical Spanish phrases in `question` (`"por encima de"`, `"por debajo de"`, `"a la izquierda de"`, `"a la derecha de"`, `"se superpone"` / `"superpuesta"`).
   * For `type == "count"`, `normalized_answer` **must be of type `int`** (`0, 1, 2, ...`), never a string `"0"`, except when the query is `"unsupported"`.
7. **Lexical Alias Expansion for LVIS (`categories.csv`):**
   * `categories.csv` already has space-separated names (`23,tennis racket`, `67,wall socket`, `81,doorknob`), but includes LVIS archaisms such as `119,fireplug` (fire hydrant), `90,spectacles` (eyeglasses), `47,television set`, `18,trousers`, `36,short pants`, `6,signboard`, `11,handle`, `110,knob`.
   * The open-vocabulary detector will use a **Visual Alias dictionary (`LVIS_PROMPT_ALIASES`)** that queries the official term plus its high-recall visual synonym in CLIP/BERT (e.g., `fireplug` + `fire hydrant` $\to$ `category_id=119`; `spectacles` + `eyeglasses` $\to$ `category_id=90`), disambiguating sibling classes (`clock` vs `watch`, `trousers` vs `jean`, `shoe` vs `boot`, `handle` vs `doorknob` vs `knob`).

---

# **3. Strategic Diagnosis: How to Surpass 0.82+ and Compete for First Place from Scratch?**

1. **Guarantee 15/15 in JSON Consistency and ~10/10 in QA from Day 1:** With the 7 normative corrections from Section 2, we ensure that the 25% non-visual/hybrid portion of the `PipelineScore` has zero leakage from schema errors, data types (`int` vs `str`), empty masks, or missing relations.
2. **Attack the Real Dataset Bottleneck (71% of Objects with `area_ratio < 0.01`):**
   * Passing the 120 categories in semantic blocks of 20–25 classes (`Chunked Prompting`) with LVIS aliases eliminates text encoder saturation in Grounding DINO / MM-Grounding-DINO.
   * Combine a **High-Resolution Global Pass** + **2×2 Overlapping Quadrant Pass (*Tiled High-Res Inference*)** to rescue tiny objects (`ring`, `watch`, `doorknob`, `handle`, `wall socket`, `fork`, `knife`, `cell phone`, `spectacles`, `sock`, `computer mouse`).
3. **Inter-Family Score Calibration + WBF with Non-Self-Collision Constraint:**
   * Align the probability scales of **OWLv2 Large** (`google/owlv2-large-patch14-ensemble`), **MM-Grounding-DINO Large**, and **Grounding DINO Base / YOLO-Worldv2-X** via quantile monotonic calibration on `CAL` before fusing with Weighted Boxes Fusion (WBF), preventing two contiguous real instances detected in the same pass from collapsing into one.
4. **Adaptive Thresholds Aware of the LVIS Federated Policy:**
   * In federated LVIS, predicting an instance only penalizes precision if the category is in `neg_category_ids` or duplicates an exhaustive positive category; conversely, omitting a positive instance always penalizes Recall (which weighs heavily on 75% of the score). Calibrate thresholds by sensitivity group (more permissive on frequent micro-objects and stricter on categories prone to background false positives like `strap`, `tag`, `pole`, `handle`, `button`).
5. **SAM2.1 Large with Box-Aware Multimask Selection (`Box-Mask Consistency`):**
   * Evaluate the 3 candidate masks from `SAM2.1 Large` (`multimask_output=True`) by combining the SAM2 score with $\text{IoU}(\text{tight\_bbox}(M_k), B_{\text{prompt}})$, avoiding under-segmentation (e.g., segmenting only the shirt when the box encloses the entire `person`).

---

# **4. Intensive 4-Day Schedule (28/10 to 31/10) Based on Jupyter Notebooks and 4 Kaggle Submissions**

| Day / Date | Master Plan Phase | Working Notebook | Experiment(s) | Main Technical Goal | Kaggle Submission (`submission.csv`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Day 1: Wednesday 28/10** | **Phase 1: Foundations, EDA, and Perfect Contract** | `01_EDA_and_Baseline_EXP01.ipynb` | **EXP-01 (Clean End-to-End Baseline)** | Stratified partition (`CAL` 80 / `VAL` 40), EDA with 4 actionable findings, 100% valid `SceneJSON` engine, deterministic QA (`questions`), and visual baseline. | **SUBMISSION #1 (28/10 night):** Validate all 80 rows, secure the 15% JSON Consistency, and measure true floor on Leaderboard. |
| **Day 2: Thursday 29/10** | **Phase 2: High-Performance Open-Vocabulary Perception** | `02_Perception_Ensemble_Tiling_EXP02.ipynb` | **EXP-02 (Chunked Prompting + LVIS Aliases + 2×2 Tiling + Calibrated WBF Ensemble)** | Optimize the 40% `Box F1` by attacking the 71% small objects ($<1\%$) and combining OWLv2 Large + MM-GDINO Large calibrated on `CAL`. | **SUBMISSION #2 (29/10 night):** Verify strong jump in `Box F1` and `Mask F1` on the public Leaderboard. |
| **Day 3: Friday 30/10** | **Phase 3: Advanced SAM2.1 Segmentation and Shielded SLM** | `03_SAM2_Multimask_and_SLM_QA_EXP03_EXP04.ipynb` | **EXP-03 (Box-Aware Multimask SAM2.1 Large) + EXP-04 (SLM Qwen2.5-3B + Guardrails)** | Optimize the 35% `Mask F1` and achieve $\ge 95\%$ fidelity on `questions_llm` across the 20 dev images with strict isolation. | **SUBMISSION #3 (30/10 afternoon/night):** Evaluate the complete integrated system (Vision + SAM2.1 + SLM) on Kaggle. |
| **Day 4: Saturday 31/10** | **Phase 4: Outstanding Extension, Final Assembly, and Report** | `04_Extension_EXP05.ipynb` and **`TPI_IA_2026P_2.ipynb`** | **EXP-05 (Inter-Family Rescue + Collective RLE Suppression + Clean-Room)** | Evaluate the Outstanding-level extension, run the 8 resilience tests, execute the full Master Notebook, and generate `REPORTE_TECNICO.md`. | **FINAL SUBMISSION #4 (31/10 before deadline):** Select the 2 best final submissions on Kaggle. |

---

# **5. Hybrid Modular Architecture (`src/` Package + Reproducible Narrative Notebooks)**

To work cleanly, quickly, and without overloading the AI Agent's context window, the project uses a structure where reusable logic resides in 7 clean modules inside `src/` (which on Day 4 are also consolidated in a self-contained manner inside `TPI_IA_2026P_2.ipynb` so it runs on Kaggle without external files):

1. `src/config_and_seeds.py`: Deterministic seed fixing (`random`, `numpy`, `torch`, `cuda`), path auto-detection (`/home/fede/Ingenieria_en_Sistemas/Inteligencia Artificial/Tp/dataset` locally vs. `/kaggle/input/...` on Kaggle), canonical `size` thresholds (`0.0035`, `0.0350`), and the `LVIS_PROMPT_ALIASES` table for the 120 categories.
2. `src/trace_logger.py`: `ExecutionTracer` class that logs per image and stage: `stage_name`, `duration_ms`, `vram_peak_mb`, `input_count`, `output_count`, `parameters_used`, `warnings`, and lineage of each object (`detector_source`, `raw_score`, `calibrated_score`, `sam2_candidate_idx`, `box_mask_iou`).
3. `src/lvis_evaluator.py`: Exact local evaluator of the LVIS federated policy (`pos_category_ids`, `neg_category_ids`, `not_exhaustive_category_ids`, unverified) with $\text{IoU} \ge 0.5$ for boxes (`xywh` $\to$ `xyxy`) and RLE masks, plus `scene_schema.json` contract validator and local `PipelineScore` estimator.
4. `src/perception_detectors.py`: Interface for open-vocabulary detectors (`OWLv2 Large`, `MM-Grounding-DINO Large`, `Grounding DINO Base`), execution with `Chunked Prompting` + LVIS aliases, overlapping 2×2 *Tiling* inference for micro-objects, monotonic score calibration, and WBF fusion with intra-detector non-self-collision constraint.
5. `src/segmentation_sam2.py`: Inference with `SAM2.1` using `xyxy` boxes as prompts, box-aware multimask selection (`Score_final(k) = alpha * s_k + (1 - alpha) * IoU_box(k)`), non-empty mask guarantee ($\sum M \ge 1$), uncompressed RLE encoding in `column-major` order (`size: [H, W]`, `counts: [...]`), and `dominant_color` extraction in CIE LAB/HSV space over an eroded copy.
6. `src/scene_builder.py`: Deterministic construction of `SceneJSON` with the 9 mandatory fields per object (`id`, `category_id`, `confidence`, `bbox`, `mask`, `centroid`, `area_ratio`, `position`, `size`), complete and symmetric graph of `relations` (`left_of`, `right_of`, `above`, `below`, `overlaps` with safe cap `< 20000`), and `jsonschema` validation.
7. `src/qa_engine.py`:
   * **Deterministic Engine (`resolve_deterministic_qa`):** Resolves the 60 prompts of `questions_dev.jsonl` (`description`, `count`, `existence`, `position`, `size`, `relation`), extracting the spatial predicate from `question` in `relation` queries, returning `int` in `normalized_answer` for `count`, duplicate-free `evidence`, and explicit handling of `unsupported` / `insufficient_information`.
   * **Image-Blind SLM Interpreter (`resolve_llm_qa`):** Uses `Qwen/Qwen2.5-3B-Instruct` (with `do_sample=False`) receiving **exclusively** `(question_text, categories_120)` verifying isolation via `assert`, generates `interpreted_query`, applies `SemanticCompletenessGuard` to detect out-of-vocabulary modifiers or classes (`unsupported`), executes the deterministic engine on `interpreted_query`, and drafts factual descriptions verified against `SceneJSON`.

---

# **6. Detailed Specification of the 5 Controlled Experiments**

## **EXPERIMENT 1 (EXP-01): Normative Audit, Decision-Driven EDA, and End-to-End Baseline**
* **Notebook:** `notebooks/01_EDA_and_Baseline_EXP01.ipynb`
* **Hypothesis:** A baseline pipeline with a JSON contract 100% verified against `scene_schema.json`, exact LVIS federated evaluation, and deterministic resolution of the 60 prompts in `questions_dev.jsonl` guarantees the 15% Structural Consistency and establishes an auditable baseline free of data leakage.
* **Design and Steps:**
  1. **Stratified Partition (`SEED = 42`):** Split the 120 development images into `CAL` (80 images, including 10 of the 20 with QA prompts) and `VAL` (40 images, including the other 10 with QA prompts).
  2. **EDA with 4 Visualizations and Actionable Conclusions:**
     * *Plot 1 (Object Scale):* Histogram/CDF of `area_ratio` showing that 71.06% of the 1,790 instances measure $< 1\%$ of the image and the median is $0.29\%$, justifying high resolution and 2×2 *tiling*.
     * *Plot 2 (LVIS Federated Policy):* Frequency of positive categories, `neg_category_ids`, and `not_exhaustive_category_ids` per image, justifying differentiated thresholds by FP risk.
     * *Plot 3 (Density and Intra-Class Occlusion):* Distribution of instance counts per image (up to $>50$ objects) and $\text{IoU}$ between GT boxes of the same category, justifying the anti-collision rule in WBF.
     * *Plot 4 (QA Benchmark Coverage):* Analysis of the 60 prompts in `questions_dev.jsonl` (queried categories, sizes of the 5 `size` instances, distances in the 10 `relation` instances, and 50/50 balance in `existence`/`relation`).
  3. **Visual and QA Baseline:** `OWLv2 Base` (`google/owlv2-base-patch16-ensemble`) + `SAM2.1 Base+` (`multimask_output=False`) + Deterministic QA Engine (`questions` and valid structured fallback in `questions_llm`).
  4. **Output and Submission #1:** Generate and validate `submissions/submission_exp01_baseline.csv` (80 rows, 0 errors in `jsonschema`) and upload to Kaggle.

## **EXPERIMENT 2 (EXP-02): Multi-Family Open-Vocabulary Perception, LVIS Aliases, and 2×2 Tiling**
* **Notebook:** `notebooks/02_Perception_Ensemble_Tiling_EXP02.ipynb`
* **Hypotheses:**
  1. Replacing the single 120-class prompt with **20–25 class blocks (`Chunked Prompting`) with LVIS visual aliases** (`fireplug` $\leftrightarrow$ `fire hydrant`, `spectacles` $\leftrightarrow$ `eyeglasses`, etc.) prevents attention dilution and recovers classes with archaic names.
  2. Adding **2×2 Tiling inference with 25% overlap** on micro-object categories drastically raises Recall on the 71% of instances with `area_ratio < 0.01`.
  3. Fusing **OWLv2 Large** (`google/owlv2-large-patch14-ensemble`) + **MM-Grounding-DINO Large** via **Monotonic Score Calibration** on `CAL` and **WBF with Intra-Detector Non-Self-Collision Constraint** increases `Box F1` by $\ge +8$ points over Baseline EXP-01.
* **Controlled Ablations on `CAL` and Confirmation on `VAL` (keeping `SAM2.1 Base+` fixed):**
  * `EXP-02A`: Single block (120 classes) vs. 5 blocks of 24 classes + `LVIS_PROMPT_ALIASES`.
  * `EXP-02B`: Full image only vs. Full image + selective 2×2 *Tiling* for micro-categories.
  * `EXP-02C`: Individual detector (`OWLv2 Large` alone vs. `MM-GDINO Large` alone) vs. Calibrated Ensemble with Anti-Collision WBF + thresholds by LVIS sensitivity group.
* **Output and Submission #2:** Generate `submissions/submission_exp02_perception.csv` and upload to Kaggle.

## **EXPERIMENT 3 (EXP-03): Box-Aware SAM2.1 Large Segmentation (`Multimask Box-Consistency`)**
* **Notebook:** `notebooks/03_SAM2_Multimask_and_SLM_QA_EXP03_EXP04.ipynb` (Part 1)
* **Hypothesis:** Running **SAM2.1 Large** with `multimask_output=True` (3 candidates per box) and choosing the mask that maximizes $\text{Score\_final}(k) = \alpha \cdot s_k + (1 - \alpha) \cdot \text{IoU}(\text{bbox}(M_k), B_{\text{prompt}})$ (evaluating $\alpha \in \{0.60, 0.75, 1.00\}$ on `CAL`) corrects under-segmentation of composite objects and improves both `Mask F1` (35% of the score) and the accuracy of `centroid`, `area_ratio`, `position`, and `size`.
* **Controlled Design:** Frozen boxes from the winner of `EXP-02`; compare `SAM2.1 Base+ (multimask=False)` vs. `SAM2.1 Large (multimask=True + Box-Consistency + non-empty guarantee)`.

## **EXPERIMENT 4 (EXP-04): Local SLM Interpreter (`Qwen2.5-3B-Instruct`), Lexical Guardrails, and Factual Verification**
* **Notebook:** `notebooks/03_SAM2_Multimask_and_SLM_QA_EXP03_EXP04.ipynb` (Part 2)
* **Hypothesis:** A lightweight SLM runnable on Kaggle (`Qwen/Qwen2.5-3B-Instruct` in FP16, `do_sample=False`) guided by a structured prompt with the ES$\to$EN dictionary of the 120 categories, plus a post-LLM guardrail (`SemanticCompletenessGuard`) and a factual description verifier, achieves $\ge 95\%$ agreement with the deterministic engine on the 40 closed questions and 0% hallucinations on the 20 descriptions, fulfilling all **Direct Pass (Promoción)** requirements.
* **Output and Submission #3:** Generate `submissions/submission_exp04_integrated.csv` (with complete `questions` and `questions_llm` in the 20 dev rows) and upload to Kaggle.

## **EXPERIMENT 5 (EXP-05): Non-Trivial Extension for Outstanding Level (`Mask-Guided Collective Suppression & Inter-Family Consensus Rescue`)**
* **Notebook:** `notebooks/04_Extension_EXP05.ipynb`
* **Motivation and Hypotheses:**
  1. *1-to-1 Inter-Family Bipartite Consensus Rescue:* Sub-threshold proposals ($[0.65 \times \tau_c, \tau_c)$) that spatially agree ($\text{IoU} \ge 0.55$) across two distinct architectural families (`OWLv2 Large` and `MM-Grounding-DINO Large`) correspond to difficult real objects (occluded or small) rather than random noise. Rescuing them improves Recall without degrading Precision.
  2. *RLE Mask-Guided Collective Box Suppression:* When a detector predicts a group box $P_{\text{macro}}$ of class $C$ enclosing $k \ge 2$ individual instances $\{P_1, \dots, P_k\}$ of the same class $C$, and the mask $M(P_{\text{macro}})$ is $\ge 75\%$ covered by the union $\bigcup_{i=1}^k M(P_i)$, suppressing $P_{\text{macro}}$ eliminates group false positives and improves counting accuracy (`count`).
* **Mandatory Critical Evaluation (Outstanding Rubric):**
  * Ablation table on `CAL` and `VAL` separating the contribution of Component 1 (Rescue), Component 2 (Collective Suppression), and both combined (`ΔTP`, `ΔFP`, `ΔFN`, `ΔBox F1`, `ΔMask F1`, `ΔQA Count Accuracy`, `Δms/img`, `ΔVRAM`).
  * Critical analysis documenting what it improves, **what limitations it retains** (e.g., when both families miss an extreme micro-object or when two contiguous objects are merged by SAM2 into the same mask), and what computational cost it introduces.

---

# **7. Operational Resilience Suite and Pre-Submission Audit (`Clean-Room Verification`)**

Before every submission to Kaggle and at the conclusion of the final notebook `TPI_IA_2026P_2.ipynb`, `run_full_verification_suite(submission_path)` must be automatically executed, validating these **8 invariants with `assert`**:

1. **Test 1 — Blind Inference (`Zero-GT Leakage Test`):** Execute `infer_scene(image_path)` on a synthetic/external image while temporarily renaming `annotations.json`, proving that inference and the QA engine operate without touching the ground truth.
2. **Test 2 — Visual Edge Cases (`Edge-Case Stress Test`):**
   * *Empty scene (0 detections):* Verify that `objects = []` and `relations = []` validate against `scene_schema.json`, `count` returns `0` (`int`) with `evidence = []`, `existence` returns `"no"`, `position` and `size` return `"insufficient_information"`, and `description` reports the absence of objects without `ZeroDivisionError`.
   * *Hyper-dense scene ($>100$ objects):* Verify that `objects` truncates by confidence to `maxItems: 300` and `relations` to `< 20000` in $< 2$ seconds.
3. **Test 3 — Symmetry and Topological Invariance of `relations`:** Verify across all 80 rows that `left_of(A, B) <=> right_of(B, A)`, `above(A, B) <=> below(B, A)`, and `overlaps(A, B) <=> overlaps(B, A)`.
4. **Test 4 — Evidence and Type Integrity in QA (`Evidence & Type Traceability Test`):**
   * In the 20 dev rows, verify that `questions` and `questions_llm` have exactly the 3 prompts corresponding to that image (`len == 3`, total 60 prompts across the 20 rows).
   * Verify that for `type == "count"` (unless `"unsupported"`), `isinstance(q["normalized_answer"], int)` is `True`.
   * Verify that every ID in `evidence` exists in `SceneJSON["objects"]`, has no duplicates, and is `[]` when `normalized_answer` is `0`, `"no"`, `"unsupported"`, or `"insufficient_information"`.
5. **Test 5 — `scene_schema.json` Compliance (`JSON Schema Compliance`):** Execute `jsonschema.validate(instance=scene_json, schema=scene_schema)` across all **80 rows** with **0 errors**.
6. **Test 6 — Column-Major RLE Mask Integrity and `area_ratio > 0`:** Across all instances in the 80 rows, check that `mask["size"] == [H, W]`, `sum(mask["counts"]) == H * W`, `len(mask["counts"]) >= 1`, `0 < obj["area_ratio"] <= 1`, and that column-major decoding yields $\ge 1$ active pixel.
7. **Test 7 — Exact Structure of `submission.csv`:** Verify exact columns `image_id,SceneJSON`, exact **80 rows** in the same ID order as `sample_submission.csv` (`test_0001`..`test_0060` followed by the 20 `tpi_XXXX`), with no `NaN` or improperly escaped JSON.
8. **Test 8 — Operational Budget on Kaggle GPU:** Confirm that full execution of the final notebook consumes $< 14\text{ GB}$ of VRAM and finishes comfortably within a single Kaggle GPU session.

---

# **8. Step-by-Step Execution Plan in Claude Code (Based on Jupyter Notebooks, Per-Task Prompts, and Effort Level)**

## **8.1 Golden Rules for Operating Claude Code Without Overloading Tokens**

1. **One Task = One Clean Session (`/clear` between tasks):** Upon finishing each task and verifying its outputs, run `/clear` in Claude Code before pasting the next task's Prompt. Since state is persisted in files (`src/`, `notebooks/`, `artifacts/`, `submissions/`), Claude Code does not need to carry over previous conversation history.
2. **Hybrid Notebook + Clean Modules Approach:**
   * Helper modules (`src/*.py`) encapsulate tested pure functions to avoid repeating 500 lines in every cell.
   * **Jupyter Notebooks (`notebooks/01_*.ipynb` to `04_*.ipynb` and the final `TPI_IA_2026P_2.ipynb`)** execute experiments step by step, interleaving **executed code cells with saved outputs + technical analysis Markdown cells** (fulfilling the course's *Data Storytelling* and traceability rule).
3. **Configuring Effort Level in Claude Code:**
   * In Claude Code you can adjust effort with `/effort low`, `/effort medium`, `/effort high`, or `/effort max` (or by including the keyword `think` / `think hard` / `ultrathink` in the prompt). Below is the exact level for each of the **8 Atomic Tasks**.

---

## **8.2 Sequence of the 8 Atomic Tasks and Ready-to-Copy-and-Paste Prompts for Claude Code**

### **TASK 1 (Day 1 — Morning): Project Foundations, Contractual Audit, and Exact Federated LVIS Evaluator**
* **Effort Level in Claude Code:** **`Medium` (`/effort medium`)** — Deterministic engineering task involving contracts, schema parsing, and well-specified geometric metrics.
* **Files created:** `src/config_and_seeds.py`, `src/trace_logger.py`, `src/lvis_evaluator.py`, `src/scene_builder.py`, `artifacts/split_cal_val.json`.
* **Prompt for Claude Code:**

```text
Act as a Senior Computer Vision Engineer and work in the current directory `/home/fede/Ingenieria_en_Sistemas/Inteligencia Artificial/Tp/TPDe0`.
First read the file `Plan Maestro Integral desde Cero — TPI Comprensión Visual de Escenas (Primavera 2026).md` (Sections 1, 2, 5, and 7) and the normative files of the official dataset in `/home/fede/Ingenieria_en_Sistemas/Inteligencia Artificial/Tp/dataset/` (`scene_schema.json`, `sample_submission.csv`, `tpi_ia_2026_competition_public/categories.csv`, `tpi_ia_2026_competition_public/test_images.csv`, `tpi_2026_qa_public/questions_dev.jsonl`, and the structure of `development/annotations.json`).

Execute ONLY TASK 1 (without implementing heavy models or inference notebooks yet):
1. Create `src/config_and_seeds.py` with:
   - Deterministic seed fixing (`SEED = 42` in `random`, `numpy`, `torch`).
   - Dataset path auto-detection (local at `/home/fede/Ingenieria_en_Sistemas/Inteligencia Artificial/Tp/dataset` vs Kaggle at `/kaggle/input/...`).
   - Loading of the 120 categories from `categories.csv` and the `LVIS_PROMPT_ALIASES` dictionary mapping each `category_id` (1..120) to its official name and clean English visual synonyms for classes with archaic or ambiguous LVIS names (`fireplug` -> `fire hydrant`, `spectacles` -> `eyeglasses`, `television set` -> `tv`, `trash can` -> `garbage can`, `doorknob`, `wall socket`, `flowerpot`, `signboard`, etc.).
   - Canonical `size` thresholds based on `area_ratio`: `small < 0.0035`, `0.0035 <= medium < 0.0350`, `large >= 0.0350`.
   - Reproducible stratified partition (`SEED = 42`) of the 120 images in `development/annotations.json` into `CAL` (80 images, including 10 of the 20 images present in `questions_dev.jsonl`) and `VAL` (40 images, including the other 10 images from `questions_dev.jsonl`), saving the result in `artifacts/split_cal_val.json`.
2. Create `src/trace_logger.py` with the `ExecutionTracer` class logging per image and per stage: `stage_name`, `duration_ms`, `vram_peak_mb`, `input_count`, `output_count`, `parameters_used`, `warnings`, and per-object traceability.
3. Create `src/scene_builder.py` with:
   - Encoding and decoding of binary masks to uncompressed RLE in `column-major` order (`size: [H, W]`, `counts: [...]` always starting with the count of zeros).
   - Strict guarantee of non-empty mask (`sum(mask) >= 1` and `0 < area_ratio <= 1`, complying with `"exclusiveMinimum": 0` in `scene_schema.json`; if a mask becomes empty, fall back to the cropped `bbox`).
   - Calculation of `centroid` `[cx, cy]`, `area_ratio`, `position` in the official 3x3 grid (`top_left`..`bottom_right`), `size` (`small`, `medium`, `large`), and `dominant_color` (on a 1px eroded copy in CIE LAB/HSV with un-eroded fallback).
   - Complete and symmetric construction of `relations` (`left_of`, `right_of`, `above`, `below` by pure centroid comparison `cx_A < cx_B`, `cy_A < cy_B` without cone clipping, plus `overlaps` if `IoU_bbox > 0` or mask intersection > 0, truncating to the top-95 objects by confidence only if `len(objects) > 95` to ensure `len(relations) < 20000`).
4. Create `src/lvis_evaluator.py` with:
   - Exact federated LVIS evaluation (`pos_category_ids`, `neg_category_ids`, `not_exhaustive_category_ids`, unverified) with `IoU >= 0.5` for boxes (converting `annotations.json` from `xywh` to `xyxy`) and RLE masks, reporting `Box Precision`, `Box Recall`, `Box F1`, `Mask Precision`, `Mask Recall`, `Mask F1`, and `LVIS Evaluation Coverage`.
   - Contractual validation suite (`validate_submission_df`) with the 7 structural tests on `scene_schema.json` and `sample_submission.csv`.
5. Run a quick terminal test verifying that `sample_submission.csv` and a `SceneJSON` built from the Ground Truth of the 20 images in `questions_dev.jsonl` pass 100% of the `scene_schema.json` and LVIS evaluator validations (obtaining `Box F1 = 1.0` and `Mask F1 = 1.0` against itself).
```

---

### **TASK 2 (Day 1 — Afternoon): Deterministic Question Engine (`questions`) + Notebook `01_EDA_and_Baseline_EXP01.ipynb` + Kaggle Submission #1**
* **Effort Level in Claude Code:** **`High` (`/effort high`)** — Requires building the exact deterministic QA engine, running the EDA with 4 plots in a real Jupyter Notebook, running Baseline `EXP-01` (120 dev + 60 test), and generating **Submission #1 (`submission_exp01_baseline.csv`)**.
* **Files created:** `src/qa_engine.py` (deterministic part), `src/perception_detectors.py` (baseline), `src/segmentation_sam2.py` (baseline), `notebooks/01_EDA_and_Baseline_EXP01.ipynb`, `submissions/submission_exp01_baseline.csv`.
* **Prompt for Claude Code:**

```text
We continue with TASK 2 of `Plan Maestro Integral desde Cero — TPI Comprensión Visual de Escenas (Primavera 2026).md` (EXP-01).
Review the modules created in `src/` and the file `/home/fede/Ingenieria_en_Sistemas/Inteligencia Artificial/Tp/dataset/tpi_2026_qa_public/questions_dev.jsonl`.

Perform the following steps:
1. Create `src/qa_engine.py` implementing the deterministic engine `resolve_deterministic_qa(scene_json, question_entry, categories_map)`:
   - Must support all 6 types in `questions_dev.jsonl`: `"description"`, `"count"`, `"existence"`, `"position"`, `"size"`, `"relation"`.
   - CRITICAL ATTENTION!:
     a) In `"count"`, `normalized_answer` MUST be a Python integer (`int`: `0, 1, 2...`), never a string `"0"`, unless the query is `"unsupported"`.
     b) In `"relation"`, `questions_dev.jsonl` provides `"class_en": [subj_cat, obj_cat]` but does NOT have a separate `predicate` field; deterministically extract the predicate from the `question` text (`"por encima de"` -> `"above"`, `"por debajo de"` -> `"below"`, `"a la izquierda de"` -> `"left_of"`, `"a la derecha de"` -> `"right_of"`, `"superpuesta"` / `"se superpone"` -> `"overlaps"`), check if the relation exists between any instance of `subj_cat` and `obj_cat` in `scene_json["relations"]` (or by comparing their centroids), return `"yes"` or `"no"` if both classes exist in the scene (or `"insufficient_information"` / `"no"` as appropriate, recording the involved instance IDs in `evidence` only when the answer is `"yes"`).
     c) In `"position"` and `"size"`, if multiple instances of the queried category exist, take the instance with highest `confidence` (or largest area) and report its `id` in `evidence`; if there are 0 instances in `scene_json["objects"]`, return `normalized_answer = "insufficient_information"` and `evidence = []`.
     d) In `"description"`, generate a deterministic factual description in Spanish from `scene_json["objects"]` and `scene_json["relations"]`, assigning the list of mentioned object IDs to `evidence`.
     e) Verify your deterministic engine by passing it the `SceneJSON` built from the Ground Truth of the 20 images in `questions_dev.jsonl` and check all 60 answers before running the detector.
2. Create the baseline versions of `src/perception_detectors.py` (using `OWLv2 Base` or `Grounding DINO Base` without ever using GT annotations during inference) and `src/segmentation_sam2.py` (using `SAM2.1` with `multimask_output=False` and non-empty mask guarantee).
3. Create and fully execute the Jupyter Notebook `notebooks/01_EDA_and_Baseline_EXP01.ipynb` (saving all outputs and inline plots), where every code cell is followed by a deep technical analysis Markdown cell:
   - Section 1: Audit of `scene_schema.json`, `sample_submission.csv`, and `CAL` (80) / `VAL` (40) partition.
   - Section 2: The 4 EDA analyses and plots specified in `EXP-01` (`area_ratio` distribution showing 71.06% < 1%, federated LVIS policy, intra-class overlap, and QA benchmark inspection).
   - Section 3: Traceable execution of Baseline `EXP-01` across the 120 dev images (`CAL` and `VAL`) and 60 test images, reporting a metrics table (`Box Precision`, `Box Recall`, `Box F1`, `Mask F1`, `LVIS Coverage`, deterministic QA, ms/img, and VRAM).
   - Section 4: Generation of `submissions/submission_exp01_baseline.csv` (80 rows; including in the 20 dev rows both `questions` and `questions_llm` initialized with a valid structured `interpreted_query` to pass 100% of the contract from Submission #1) and execution of `run_full_verification_suite("submissions/submission_exp01_baseline.csv")` with 0 errors.
```

> **Human Control Milestone #1 (Night of 28/10):** Upload `submissions/submission_exp01_baseline.csv` to Kaggle to confirm the true baseline score and verify that the 15% JSON Consistency is 100% secured.

---

### **TASK 3 (Day 2 — Morning): `EXP-02A` and `EXP-02B` — Chunked Prompting + LVIS Aliases + 2×2 Tiling for Micro-Objects**
* **Effort Level in Claude Code:** **`High` (`/effort high`)** — Requires designing and validating *Chunked Prompting* with LVIS aliases and 2×2 *Tiling* with coordinate projection to capture the 71% of objects with `area_ratio < 0.01`.
* **Files modified/created:** `src/perception_detectors.py`, `notebooks/02_Perception_Ensemble_Tiling_EXP02.ipynb` (Sections 1 and 2).
* **Prompt for Claude Code:**

```text
We start TASK 3 (first half of EXP-02 in `Plan Maestro Integral desde Cero — TPI Comprensión Visual de Escenas (Primavera 2026).md`).
Remember that 71.06% of instances in `annotations.json` have `area_ratio < 0.01` and the median is `0.00289` (`ring`, `necklace`, `watch`, `handle`, `doorknob`, `wall socket`, `fork`, `knife`, `cell phone`, `spectacles`, `sock`, `computer mouse`).

Perform the following steps:
1. In `src/perception_detectors.py`, implement:
   - `Chunked Prompting` (splitting the 120 categories into semantic blocks of 20–24 categories) integrated with `LVIS_PROMPT_ALIASES` (querying English visual aliases and remapping each detection to its official `category_id` `1..120`, with disambiguation rules for sibling classes: `clock` vs `watch`, `trousers` vs `jean`, `shoe` vs `boot`, `handle` vs `doorknob` vs `knob`).
   - `Multi-Scale Tiled Inference (2x2 Overlapping Tiles)`: an additional pass over 4 quadrants with 25% overlap for the subset of small/medium categories, projecting detected boxes to the global coordinate system `[0, W] x [0, H]` and eliminating truncated artifacts right on the internal tile border.
2. Create the Jupyter Notebook `notebooks/02_Perception_Ensemble_Tiling_EXP02.ipynb` and execute (with technical analysis Markdown cells after each result):
   - Controlled ablation `EXP-02A` on `CAL` (80 images): Single-block prompt (120 raw classes) vs. `Chunked Prompting` (5 blocks of 24 classes + `LVIS_PROMPT_ALIASES`).
   - Controlled ablation `EXP-02B` on `CAL` (80 images): Full image alone vs. Full image + selective `Tiling 2x2`, breaking down the Recall increase across small (`area_ratio < 0.0035`), medium, and large objects.
   - Save raw predictions to cache (`artifacts/cache_exp02/`) to avoid repeating GPU inference in Task 4.
```

---

### **TASK 4 (Day 2 — Afternoon): `EXP-02C` — Multi-Family Ensemble (`OWLv2 Large` + `MM-GDINO Large`), Score Calibration, Anti-Collision WBF, and Kaggle Submission #2**
* **Effort Level in Claude Code:** **`Max` (`/effort max` / `ultrathink`)** — This is the heart of the 40% `Box F1` (and a necessary condition for the 35% `Mask F1`). Requires statistical calibration on `CAL` without overfitting, WBF with intra-detector non-self-collision constraint, and bootstrap verification on `VAL`.
* **Files modified/created:** `src/perception_detectors.py`, `notebooks/02_Perception_Ensemble_Tiling_EXP02.ipynb` (Sections 3, 4, and 5), `submissions/submission_exp02_perception.csv`.
* **Prompt for Claude Code:**

```text
use ultrathink

We continue with TASK 4 (closing EXP-02 in `Plan Maestro Integral desde Cero — TPI Comprensión Visual de Escenas (Primavera 2026).md`).
The goal is to maximize `0.40 * Box F1 + 0.35 * Mask F1` by combining complementary families of open-vocabulary detectors without collapsing nearby instances and without overfitting thresholds.

Perform the following steps:
1. In `src/perception_detectors.py`, incorporate the second detector from a distinct family (combining `OWLv2 Large` `google/owlv2-large-patch14-ensemble` + `MM-Grounding-DINO Large` / `Grounding DINO Base`, and optionally `YOLO-Worldv2-X` if it adds positive diversity on `CAL`).
2. Implement **Per-Family Monotonic Score Calibration** fitted exclusively on `CAL` (80 images):
   - Since `OWLv2` produces low sigmoid scores (0.05–0.30) and `Grounding DINO` produces higher scores (0.20–0.65), map via quantile/temperature monotonic transformation the F1-optimal threshold of each detector on `CAL` to a common canonical score of `0.50` prior to fusion.
3. Implement **Weighted Boxes Fusion (WBF) with Intra-Detector Non-Self-Collision Constraint**:
   - When fusing proposals of the same category with `IoU >= 0.55`, allow fusion across different detectors or different tiles, but prevent two high-confidence boxes emitted by the same detector in the same view from fusing with each other unless there is extreme overlap (`IoU >= 0.80`), preserving contiguous instances (chairs, plates, shoes, books, people).
   - Tune post-fusion thresholds on `CAL` by LVIS sensitivity level (standard vs. micro-objects vs. categories prone to verified false positives in `neg_category_ids`).
4. Complete and execute Sections 3, 4, and 5 of `notebooks/02_Perception_Ensemble_Tiling_EXP02.ipynb`:
   - Full comparative table on `CAL` (80) and `VAL` (40) with bootstrap confidence intervals (1,000 iterations): `EXP-01 Baseline` vs `EXP-02A` vs `EXP-02B` vs `EXP-02C (Calibrated Ensemble + Anti-Collision WBF)`.
   - Visualization of examples where the ensemble rescues micro-objects and avoids false positives in `neg_category_ids`.
   - Generate `submissions/submission_exp02_perception.csv` (80 rows) using the winning detector from `EXP-02C` + `SAM2.1` and run `run_full_verification_suite("submissions/submission_exp02_perception.csv")` with 0 errors.
```

> **Human Control Milestone #2 (Night of 29/10):** Upload `submissions/submission_exp02_perception.csv` to Kaggle and verify the real increase on the public Leaderboard.

---

### **TASK 5 (Day 3 — Morning): `EXP-03` — `SAM2.1 Large` Segmentation with Box-Aware Multimask Selection (`Box-Mask Consistency`)**
* **Effort Level in Claude Code:** **`High` (`/effort high`)** — Directly optimizes the 35% `Mask F1` and refines derived geometric attributes (`centroid`, `area_ratio`, `position`, `size`).
* **Files modified/created:** `src/segmentation_sam2.py`, `notebooks/03_SAM2_Multimask_and_SLM_QA_EXP03_EXP04.ipynb` (Part 1: `EXP-03`).
* **Prompt for Claude Code:**

```text
We execute TASK 5 (`EXP-03` of `Plan Maestro Integral desde Cero — TPI Comprensión Visual de Escenas (Primavera 2026).md`).
Freeze the boxes detected by the best model from `EXP-02C` to perform a 100% controlled comparison of the segmentation stage.

Perform the following steps:
1. Update `src/segmentation_sam2.py` to run `SAM2.1 Large` (`facebook/sam2.1-hiera-large` or the largest `SAM2.1` checkpoint available within the VRAM budget) with `multimask_output=True` (obtaining 3 candidate masks `{M_1, M_2, M_3}` and their scores `{s_1, s_2, s_3}` for each box `B = [x1, y1, x2, y2]`):
   - For each candidate `M_k`, compute its tight bounding box `B(M_k)` and the overlap `IoU_box(k) = IoU(B(M_k), B)`.
   - Evaluate the selection function `Score_final(k) = alpha * s_k + (1 - alpha) * IoU_box(k)` for `alpha in {0.60, 0.75, 1.00}` on `CAL`.
   - Apply safe morphological cleaning: remove isolated connected components smaller than 15 pixels ONLY IF it is not the sole connected component of the mask (invariably guaranteeing `sum(mask) >= 1` and `area_ratio > 0`).
2. Create the first half of the Jupyter Notebook `notebooks/03_SAM2_Multimask_and_SLM_QA_EXP03_EXP04.ipynb` and execute:
   - Controlled comparison on `CAL` and `VAL`: `SAM2.1 Base+ (multimask=False)` vs `SAM2.1 Large (multimask=False)` vs `SAM2.1 Large (multimask=True + Box-Consistency)`.
   - Measure the impact on `Mask Precision`, `Mask Recall`, `Mask F1`, centroid deviation from GT, and classification accuracy of `size` (`small`, `medium`, `large`) and `position` (3x3).
   - Include a side-by-side comparative figure showing how `Multimask Box-Consistency` corrects under-segmentation on composite objects (e.g., `person`, `bicycle`, `table`, `sofa`) and its explanation in Markdown.
```

---

### **TASK 6 (Day 3 — Afternoon): `EXP-04` — Local SLM Interpreter (`Qwen2.5-3B-Instruct`), Semantic Guardrails, Factual Description Verifier, and Kaggle Submission #3**
* **Effort Level in Claude Code:** **`High` (`/effort high`)** — Requires cryptographic/contractual isolation of the SLM, ES$\to$EN lexical disambiguation across the 120 categories, guard against silent degradation (`unsupported`), and factual verification of descriptions.
* **Files modified/created:** `src/qa_engine.py`, `notebooks/03_SAM2_Multimask_and_SLM_QA_EXP03_EXP04.ipynb` (Part 2: `EXP-04`), `submissions/submission_exp04_integrated.csv`.
* **Prompt for Claude Code:**

```text
We continue with TASK 6 (`EXP-04` of `Plan Maestro Integral desde Cero — TPI Comprensión Visual de Escenas (Primavera 2026).md`).
The goal is to complete the LLM mode (`questions_llm`) required for Direct Pass (Promoción) and Outstanding (Sobresaliente), achieving >= 95% functional equivalence with the deterministic engine on the 20 images in `questions_dev.jsonl` and 100% factual fidelity in descriptions.

Perform the following steps:
1. In `src/qa_engine.py`, implement the `SLMQuestionInterpreter` module using `Qwen/Qwen2.5-3B-Instruct` in FP16 with deterministic decoding (`do_sample=False`, `temperature=None`, `top_p=None`):
   - **Strict Isolation (Validity Rule):** The function `interpret_question_with_llm(question_text: str, categories_120: list[str]) -> dict` must verify via `assert` that it receives EXCLUSIVELY the `question_text` string and the list of 120 English categories, with no access to `SceneJSON` or `class_en`/`class_es`/`type`/`allowed_answers` from `questions_dev.jsonl`.
   - **Translation to `interpreted_query`:** The SLM must return a validated intermediate JSON with `{"operation": ..., "target_category": ..., "secondary_category": ..., "predicate": ..., "expected_output": ..., "unsupported_reason": ...}`.
   - **Post-LLM Guardrail (`SemanticCompletenessGuard`):** Implement a deterministic verifier in Python that audits `interpreted_query` against `question_text`:
     a) Fallback ES->EN lexical mapping for the 120 categories (including contextual disambiguation of Spanish polysemy, e.g., `"mango"` of a door/utensil -> `handle` vs out-of-vocabulary fruit -> `unsupported`; `"banco"` seat -> `bench` vs financial institution -> `unsupported`), without ever using rules by `question_id` or `image_id`.
     b) Detection of modifiers or attributes unsupported by the engine (e.g., if a question asks for a non-existent attribute such as broken material, commercial brand, or complex action outside the schema), forcing `operation = "unsupported"`, `normalized_answer = "unsupported"`, and `evidence = []`.
   - **Execution and Factual Verbalization:** Run the EXACT SAME Python deterministic engine over `interpreted_query` and `SceneJSON` to obtain `normalized_answer` (remembering that `count` takes an `int`), `evidence`, and `confidence`. For `type == "description"`, pass the SLM only the structured summary of facts from `SceneJSON` and verify in Python that the generated Spanish description does not mention quantities or objects non-existent in `SceneJSON`, populating `evidence` with all supporting `object_id`s.
2. Complete Part 2 of `notebooks/03_SAM2_Multimask_and_SLM_QA_EXP03_EXP04.ipynb`:
   - Evaluate `questions` and `questions_llm` on the 60 real prompts of `questions_dev.jsonl` AND on an additional battery of **12 synthetic stress queries** (4 ambiguous, 4 with classes outside the 120 categories, and 4 with unsupported attributes) demonstrating explicit handling of `unsupported` and `insufficient_information`.
   - Generate `submissions/submission_exp04_integrated.csv` (80 complete rows with `EXP-02C` + `EXP-03` + `EXP-04`) and run `run_full_verification_suite("submissions/submission_exp04_integrated.csv")` with 0 errors.
```

> **Human Control Milestone #3 (Night of 30/10):** Upload `submissions/submission_exp04_integrated.csv` to Kaggle to evaluate the complete system (with `EXP-02`, `EXP-03`, and `EXP-04`).

---

### **TASK 7 (Day 4 — Morning): `EXP-05` — Outstanding-Level Extension (`Mask-Guided Collective Suppression & Inter-Family Consensus Rescue`) and Forensic Error Analysis**
* **Effort Level in Claude Code:** **`Max` (`/effort max` / `ultrathink`)** — This is the differential requirement for the **Outstanding (Sobresaliente)** grade and for squeezing out the final tenths of `PipelineScore` on Kaggle. Requires rigorous ablation of its two components and critical analysis of limitations and error propagation.
* **Files modified/created:** `src/perception_detectors.py`, `src/segmentation_sam2.py`, `notebooks/04_Extension_EXP05.ipynb`, `submissions/submission_exp05_sobresaliente.csv`.
* **Prompt for Claude Code:**

```text
use ultrathink

We execute TASK 7 (`EXP-05`: Extension for Outstanding Level in `Plan Maestro Integral desde Cero — TPI Comprensión Visual de Escenas (Primavera 2026).md`).
To reach the Outstanding level, we need to implement, measure, and critically analyze the `Mask-Guided Collective Suppression & Inter-Family Consensus Rescue` extension, in addition to the end-to-end error propagation analysis.

Perform the following steps:
1. Implement in a modular and configuration-togglable manner (`enable_consensus_rescue: bool`, `enable_collective_suppression: bool`):
   - **Component 1 (1-to-1 Inter-Family Bipartite Consensus Rescue):** Prior to individual threshold cutoff, identify pairs of sub-threshold proposals (`score in [0.65 * tau_c, tau_c)`) of the same category emitted by two distinct families (`OWLv2 Large` and `MM-Grounding-DINO Large`) with `IoU >= 0.55` that do not collide (`IoU < 0.45`) with already accepted detections. Rescue the fused instance, recording `"rescued_by_inter_family_consensus": True` in the trace.
   - **Component 2 (RLE Mask-Aware Collective Box Suppression):** After obtaining masks from `SAM2.1 Large`, if a prediction `P_macro` of category `C` contains within its box `k >= 2` smaller instances `{P_1, ..., P_k}` of the same category `C`, and the active area of mask `M(P_macro)` is `>= 75%` covered by the union of masks `U M(P_i)`, suppress `P_macro` as a redundant group box and log the suppression in `ExecutionTracer`.
2. Create and execute the Jupyter Notebook `notebooks/04_Extension_EXP05.ipynb` with a complete technical narrative:
   - **Quantitative Ablation Study on `CAL` (80) and `VAL` (40):** Compare:
     (a) System without extension (`EXP-04`),
     (b) Component 1 only (`+Consensus Rescue`),
     (c) Component 2 only (`+Collective Mask Suppression`),
     (d) Full Extension (`EXP-05`).
     Report `TP`, `FP`, `FN`, `Box Precision`, `Box Recall`, `Box F1`, `Mask F1`, `QA Count Accuracy`, additional latency (`ms/img`), and VRAM.
   - **Critical Analysis (Outstanding Rubric Requirement):** Document with real visual figures:
     * 2 success cases (an object rescued by inter-family consensus and a collective box suppressed by RLE masks).
     * 1 edge/failure case where the extension FAILS to solve the problem (honestly explaining what limitations it retains and what computational cost it introduces).
   - **End-to-End Forensic Error Propagation Analysis:** Select 3 real scenes (1 successful, 1 ambiguous, 1 failed) and trace with `ExecutionTracer` how an error in `Perception` or `Segmentation` propagates to `Representation` (`centroid`, `position`, `size`, `relations`) and `Reasoning` (`questions` / `questions_llm`).
   - Generate `submissions/submission_exp05_sobresaliente.csv` (and `submission.csv` in the root) and verify all 8 tests of `run_full_verification_suite()` with 0 errors.
```

---

### **TASK 8 (Day 4 — Afternoon): Consolidation of the Master Notebook `TPI_IA_2026P_2.ipynb`, Clean-Room Run, and Generation of `REPORTE_TECNICO.md`**
* **Effort Level in Claude Code:** **`High` (`/effort high`)** — Requires consolidating all work into the single mandatory submission notebook (`TPI_IA_2026P_2.ipynb`), running it end-to-end with all outputs saved, and writing `REPORTE_TECNICO.md` 100% consistent with the notebook cells.
* **Files created:** `TPI_IA_2026P_2.ipynb`, `REPORTE_TECNICO.md`, `submission.csv`.
* **Prompt for Claude Code:**

```text
We execute the FINAL TASK 8 of `Plan Maestro Integral desde Cero — TPI Comprensión Visual de Escenas (Primavera 2026).md` (Consolidation of Official Deliverables).
Review the submission requirements on pages 10 and 11 of the Official Guidelines:
- Single reproducible and self-contained notebook `TPI_IA_2026P_2.ipynb` with all outputs saved, fixed seeds, without depending on hidden local files outside the official dataset or public downloads performed by the notebook itself.
- Technical report in Markdown `REPORTE_TECNICO.md` matching the numbers, tables, and plots of the notebook exactly.
- Valid 80-row `submission.csv` file.

Perform the following steps:
1. Build the self-contained master notebook `TPI_IA_2026P_2.ipynb` (cleanly incorporating the classes from `src/` inside the corresponding sections of the notebook so it does not depend on external modules when uploaded to Kaggle), organized with explanatory Markdown cells before and after every code block across 9 sections:
   - Section 0: Configuration, Seed Fixing, and Environment Auto-Detection (Kaggle / Local).
   - Section 1: Dataset Exploration and Characterization (EDA with the 4 visualizations and derived decisions).
   - Section 2: Stage 1 — Open-Vocabulary Perception (`Chunked Prompting`, `LVIS_PROMPT_ALIASES`, `Tiling 2x2`, Inter-Family Calibration, and Anti-Collision WBF).
   - Section 3: Stage 2 — Instance Segmentation with `SAM2.1 Large` (`Multimask Box-Consistency` and Column-Major RLE).
   - Section 4: Stage 3 — `SceneJSON` Construction, Derived Attributes, and Spatial Relations Graph.
   - Section 5: Stage 4 — Traceability (`ExecutionTracer`) and Execution Audit.
   - Section 6: Stage 5 — Question Answering and Description: Deterministic Mode (`questions`) and SLM Interpreter Mode (`questions_llm` with `Qwen2.5-3B-Instruct`, `SemanticCompletenessGuard`, and Factual Verifier).
   - Section 7: Controlled Experimental Process (`EXP-01` to `EXP-04`) and Outstanding-Level Extension (`EXP-05`: Quantitative Contribution, Limitations, and Cost) + Error Propagation Analysis.
   - Section 8: Generation of `submission.csv` (80 rows) and Execution of the 8 Tests of `run_full_verification_suite("submission.csv")`.
2. Execute the master notebook from start to finish verifying that **all cells have their outputs, tables, and plots saved** and 0 cells have errors.
3. Generate `REPORTE_TECNICO.md` following the 6 official sections (1. Cover Page, 2. Dataset Analysis, 3. Methodology, 4. Results, 5. Discussion and Error Analysis, 6. Conclusions and Future Work), verifying that every cited metric matches the notebook outputs 100%.
```