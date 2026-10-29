---
## Section 1 — Normative audit: `scene_schema.json`, `sample_submission.csv` and the CAL/VAL partition

Before measuring anything we pin down the contract the Kaggle metric consumes. Every later number in this notebook is meaningless if a single row of the submission fails schema validation (the whole row would lose its 15% JSON-consistency contribution and its detections).
