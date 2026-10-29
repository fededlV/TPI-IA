**Qualitative cases.**

1. **Micro-objects rescued (`tpi_0104`, VAL).** EXP-02C matches four small GT instances that EXP-02B misses: sunglasses (`area_ratio` 0.0004), a taillight (0.0006), short pants (0.0026) and a necklace (0.0030).
   - Over VAL, EXP-02C gains **25 small instances and loses 20**. Over DEV the balance is +86 / −56.
   - The categories gained most on DEV are person, car, shoe, wheel, chair, curtain, pillow, hat and book, mostly crowded scenes where the families corroborate each other.
2. **Verified false positives removed (`tpi_0023`, VAL).** EXP-02B predicts three `napkin` boxes on an image where `napkin` is a verified-negative category. EXP-02C keeps none: no yellow boxes.
   - Scale matters here. Verified-negative FPs are only ~1% of all box FPs (DEV: 6 for EXP-02B, 3 for EXP-02C).
   - The real precision gain is elsewhere: total box FPs on DEV fall from **629 to 496 (−21%)**, almost all of them unmatched or duplicate boxes in exhaustively annotated positive categories.
3. **Adjacent instances kept apart (third panel).** **No case exists.** No DEV GT instance changes its match status because of the self-collision rule (Section 4 above), so the selector fell back to an arbitrary image (`tpi_0120`) with nothing highlighted. The empty panel is the visual form of the negative result above.
