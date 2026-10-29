# EXP-01 — Normative Audit, Decision-Driven EDA and End-to-End Baseline

**TPI Visual Scene Understanding — UTN Artificial Intelligence, Spring 2026 — Group 2**

**Hypothesis (EXP-01).** A baseline pipeline whose `SceneJSON` contract is verified against `scene_schema.json`, evaluated with the exact LVIS federated policy, and whose 60 `questions_dev.jsonl` prompts are resolved by a deterministic engine, secures the 15% JSON-consistency term and establishes an auditable, leakage-free reference point for every later experiment.

**Pipeline under test** (`Perception → Segmentation → Representation → Reasoning`):

| Stage | Component | Module |
|---|---|---|
| Perception | OWLv2 Base (`google/owlv2-base-patch16-ensemble`), prompts = the 120 official names, per-class NMS 0.5 | `src/perception_detectors.py` |
| Segmentation | SAM2.1 Base+ (`facebook/sam2.1-hiera-base-plus`), one box prompt per detection, `multimask_output=False`, non-empty mask guarantee | `src/segmentation_sam2.py` |
| Representation | Column-major RLE, centroid, `area_ratio`, 3×3 position, size, dominant color, symmetric relations | `src/scene_builder.py` |
| Reasoning | Deterministic QA engine (`questions`) + rule-based text interpreter feeding the same engine (`questions_llm`) | `src/qa_engine.py` |
| Evaluation | LVIS federated evaluator (IoU ≥ 0.5) + 8-test contract suite | `src/lvis_evaluator.py` |

**Controlled-experiment discipline.** EXP-01 deliberately uses *no* LVIS prompt aliases, *no* chunked prompting and *no* tiling: those are the variables of EXP-02A/B, and introducing them here would make the later ablations uninterpretable. The only tuned parameter is a single global score threshold, selected on `CAL` and only *reported* on `VAL`.

**Epistemic convention used throughout.** *Observed* = detector boxes/scores and SAM2 masks; *derived* = `area_ratio`, `centroid`, `position`, `size`, `dominant_color`, `relations`; *unknown* = answered `unsupported` / `insufficient_information`. `annotations.json` is read in this notebook **only** for EDA and evaluation — inference functions never receive it (verified by Test 1 of the suite with an audit hook on `open()`).

**Compute note.** This notebook was executed on a CPU-only workstation (no CUDA device). Latencies are therefore CPU latencies and VRAM is reported as *N/A (CPU)*; suite Test 8 (VRAM budget) stays `SKIPPED` until the Kaggle GPU run.
