**Decision — configuration carried into EXP-02B.**

- The pre-declared rule (maximum CAL Box F1 over the 8 global arms) selects **OWLv2 Base + `chunked_raw` at threshold 0.175** (F1 0.6596).
- Its F1 advantage over `single_raw` (+0.002) is not significant, so the choice is justified on two other grounds:
  1. It is the only prompting change with a **significant recall gain** and no significant precision cost.
  2. With OWLv2 it is **free**: same forward pass, same latency.
- Aliases are dropped from the default configuration (significant precision loss). Large is not carried forward (significantly worse).
- EXP-02B therefore isolates tiling on top of `base / chunked_raw`. Section 2.5 checks that the tiling effect does not depend on this choice.
