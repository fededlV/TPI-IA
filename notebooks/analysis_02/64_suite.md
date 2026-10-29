**Contract suite: 8/8 PASS, 0 FAIL, on the GPU.**
- **Test 1 (blind inference)** ran the *live* ensemble on a synthetic image: OWLv2 Base + MM-GDINO Large through `EnsembleDetector`, then SAM2.1. An audit hook confirmed that `annotations.json` was never opened.
- **Test 8 (VRAM):** peak 8,326 MB against the 14,336 MB budget.
- This is the first submission to pass all 8 tests. Submission #1 had Test 8 SKIPPED because it ran on CPU.
