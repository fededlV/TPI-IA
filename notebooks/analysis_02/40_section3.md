## Section 3 — EXP-02C, part 1: a second detector family and per-view score calibration

**Why a second family.** EXP-02B ended with precision as the bottleneck: tiling's recall gain was cancelled by uncorroborated tile false positives. A box that two *architecturally different* detectors agree on is far less likely to be noise than a box only one of them proposes. That only helps if the second family finds **different** objects, so diversity is measured before any fusion.

**The second family: MM-Grounding-DINO Large** (`openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg`, Swin-L, zero-shot checkpoint).
- It is a DETR-style detector with 900 learned queries and a BERT text encoder fused with the image (early fusion). OWLv2, by contrast, is a CLIP ViT that scores every patch against each text embedding independently.
- `*_large_all` was excluded: its pre-training includes COCO/LVIS-family data, which would weaken the zero-shot claim.

**Prompting.**
- The 120 classes are sent as the same 5 semantic blocks × 24 used for OWLv2: one forward pass per block and view.
- For GDINO the chunking is **mandatory**, not an optimisation. The 120 names in one prompt need about 300 BERT tokens, above the 256-token limit. The blocks use 56–64 tokens each.
- Each category's score is the max over its own tokens (char-offset mapping, verified against the real tokenizer).

**Views.** The same 2×2 tiles as EXP-02B: 5 views × 5 blocks = **25 forward passes per image**.
