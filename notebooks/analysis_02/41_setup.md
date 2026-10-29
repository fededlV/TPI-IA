**Cache provenance.**
- Kaggle Tesla T4, FP32, `torch 2.11.0+cu128`. 180 images (120 dev + 60 test), score floor 0.10.
- Raw output: 74 global and 192 tile detections per image.
- The text budget is respected with a wide margin: at most 64 tokens per block.
- The run was resumed once after a session interruption. The cache is complete (asserted above), but the timing trace covers only 150 of the 180 images.
