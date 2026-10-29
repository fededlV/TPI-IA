---
## Section 4 — Submission #1 (`submissions/submission_exp01_baseline.csv`) and contract verification

The submission has exactly the 80 rows of `sample_submission.csv`, in the same order: 60 test scenes and the 20 development scenes of the QA benchmark. Development rows are built from **predicted** scenes (never from GT) and carry both `questions` (deterministic engine with the structured fields) and `questions_llm` (rule-based interpreter that reads only the raw question text and the 120-category vocabulary, emitting a structured `interpreted_query` that is then executed by the same deterministic engine — the placeholder for the EXP-04 SLM).
