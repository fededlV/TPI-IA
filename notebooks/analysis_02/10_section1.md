## Section 1 — EXP-02A: Single Prompt (120 raw classes) vs. Chunked Prompting (5 × 24) + LVIS Aliases

**A mechanistic point fixes how this ablation should be read.**
- OWLv2's class head scores each text query **independently**: a dot product between the box embedding and the normalised query embedding, with a shift/scale that depends only on the image (`modeling_owlv2.py`). Adding or removing queries never changes another query's score.
- So the "attention dilution" argument behind chunked prompting does not apply to OWLv2. It applies to text encoders that read all classes in one sequence, such as Grounding DINO / MM-GDINO and their 256-token limit (T4).
- For OWLv2, chunking changes exactly one thing: the label of each box is the argmax **within each block**, so a box can carry up to 5 labels (one per block) instead of a single one.
- Because scores do not interact, **one forward pass over the 183 alias queries is mathematically identical to five chunked passes**. The runner therefore extracts a full 2×2 factorial from a single image encoding:

| variant | blocks | queries per category |
|---|---|---|
| `single_raw` (= EXP-01) | 1 × 120 | official name |
| `single_alias` | 1 × 120 | max over official name + aliases |
| `chunked_raw` | 5 × 24 | official name |
| `chunked_alias` (master-plan treatment) | 5 × 24 | max over official name + aliases |

This separates the effect of chunking from the effect of aliases, which a plain "single block vs. 5 blocks + aliases" comparison would confound.
