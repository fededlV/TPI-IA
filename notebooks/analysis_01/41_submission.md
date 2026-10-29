**Analysis — Submission #1 file.**

- `submissions/submission_exp01_baseline.csv`: **80 rows** in the exact order of `sample_submission.csv`, columns `image_id,SceneJSON`, ~20 MB (dominated by the uncompressed column-major RLE masks and the full relation graphs, both mandated by the schema).
- Each of the 20 development rows has exactly 3 entries in `questions` and 3 in `questions_llm` (60 + 60 in total), including the `description` prompt. Test rows carry no QA fields (allowed by the README).
- The `questions_llm` example shows the contract of the LLM mode: a typed `normalized_answer` (`0`, a Python `int`, here an honest undercount caused by the detector), empty `evidence` for a zero count, and an `interpreted_query` documenting *how* the question text was understood (`source: rule_based_fallback`, type, resolved classes and ids, predicate). Nothing in it was copied from the structured fields of `questions_dev.jsonl`.
- `ensure_ascii=False` keeps Spanish answers human-readable inside the CSV; the CSV writer handles quoting, which Test 7 re-validates by parsing every row back.
