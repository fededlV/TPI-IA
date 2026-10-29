**Analysis 2 — LVIS federated label state (Plot 2a/2b).**

- **Per image, the vocabulary splits into:** 6.37 positive categories on average (FN risk: every missed instance costs recall), 0.98 verified-absent categories (`neg_category_ids`, the *only* source of category-level false positives), 0.39 non-exhaustive categories (extra predictions are ignored), and **112.65 unverified categories** whose predictions are neither rewarded nor penalised.
- **Only 6.1% of the vocabulary is evaluable per image** ((pos + neg) / 120). Consequently, *LVIS Evaluation Coverage* (fraction of our predictions that fall in an evaluable category) is expected to be low for an open-vocabulary detector prompted with all 120 classes, and a low coverage is **not** an error per se — it measures how much of our output the metric actually sees.
- **Negatives are sparse but concentrated:** 74 of 120 images have at least one `neg` id, at most 4 per image. The most frequent verified-absent classes (`sofa` 6, `clock` 5, `ski` 4, `napkin`, `flag`, `handbag`, `baseball glove`, `computer monitor` 3 each) are exactly the classes an open-vocabulary detector hallucinates in indoor/sports scenes; these are the FP hot spots.
- **Non-exhaustive classes** (`strap`, `boat`, `car`, `book`, `street sign`, …) are crowd-like categories where annotators stopped early; over-predicting them costs nothing beyond the matched instances.

**Decisions.**
1. Under this policy precision is only hurt by (a) duplicates/unmatched boxes in exhaustive positive classes and (b) predictions in `neg` classes, while recall is hurt by every miss. The optimal operating point is therefore *more permissive* than on an exhaustively labelled dataset — the CAL threshold sweep in Section 3 quantifies this.
2. EXP-02C introduces per-group thresholds: stricter for classes that frequently appear as verified negatives, more permissive for micro-object classes.
