**Analysis — which categories are tiled.**

- **Rule.**
  - A category is tiled when its **CAL** GT median `area_ratio` is below the medium/large boundary (0.035). Categories absent from CAL are included by default.
  - This is a training-side hyperparameter, exactly like the CAL score threshold: it is fixed once for the whole dataset, never chosen per image and never from the image being predicted, so it does not leak ground truth into inference. It is frozen in `artifacts/cache_exp02/tiling_categories.json` for T4.
- **Result.** 106 of 120 categories are tiled. The 14 excluded ones are the obvious large-scale classes (`bus`, `bed`, `train`, `horse`, `blanket`, `motorcycle`, `curtain`, `surfboard`, `cat`, `sofa`, `toilet`, `refrigerator`, `computer monitor`) **and `person`** (median 0.042).
- **Why excluding `person` matters.** A person seen through a tile is a partial body (torso, legs) that OWLv2 happily labels `person`. These partial boxes do not match the full-body GT and, since `person` is almost always an exhaustive positive, every one of them is a federated false positive. The next table shows this effect.
