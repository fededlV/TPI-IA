# Running the experiments on a GPU (Colab or Kaggle)

The `src/` modules pick the device automatically (`config_and_seeds.resolve_device()`: `TPI_DEVICE` env var, else
CUDA when available, else CPU), and `ExecutionTracer` records the CUDA peak memory, so the same code runs locally
on CPU (EXP-01) and on a T4 GPU (EXP-02 onwards). Detection/segmentation caches record the device they ran on.

## Kaggle (recommended: dataset already mounted, same environment as the final notebook)

1. Open a notebook from the competition page, enable **GPU T4** and **Internet** in the settings.
2. First cell:
   ```python
   !git clone -b <branch> https://github.com/fededlV/TPI-IA.git
   %cd TPI-IA
   !grep -v "^torch" requirements.txt | pip install -q -r /dev/stdin   # keep Kaggle's CUDA torch
   ```
3. The dataset root is auto-detected under `/kaggle/input/**/scene_schema.json`; no path setup is needed.

## Google Colab

1. *Runtime → Change runtime type → T4 GPU*.
2. Upload the dataset folder (the one containing `scene_schema.json`, ~75 MB) to Google Drive, then:
   ```python
   from google.colab import drive; drive.mount("/content/drive")
   !git clone -b <branch> https://github.com/fededlV/TPI-IA.git
   %cd TPI-IA
   !grep -v "^torch" requirements.txt | pip install -q -r /dev/stdin   # keep Colab's CUDA torch
   import os; os.environ["TPI_DATASET_DIR"] = "/content/drive/MyDrive/<path-to>/dataset"
   ```

## Then, in either environment

```python
!python scripts/run_exp01_inference.py detect
!python scripts/run_exp01_inference.py segment --threshold 0.175
!python scripts/build_notebook_01.py --execute
```

Copy `artifacts/cache_*` and the executed notebook back to the repository (or commit from the runtime) so results
stay traceable. Never commit `kaggle.json` or any token.

## EXP-02A/B (T3): raw detection caches on GPU

```python
!python scripts/run_exp02_inference.py --backbone base --limit 2   # smoke test (~1 min)
!python scripts/run_exp02_inference.py --backbone base             # OWLv2 Base, 180 images x 5 views
!python scripts/run_exp02_inference.py --backbone large            # OWLv2 Large (reused by T4)
```

Each image gets 5 forward passes (global + 2x2 tiles) over 183 text queries (120 official names + aliases); the four
EXP-02A variants come out of the same pass. On CPU this takes ~70 s/img for Base (~3.5 h for 180 images), so run it
on a T4. The runner resumes from `artifacts/cache_exp02/<backbone>/detections_<variant>.json.gz` (~7 MB per backbone),
so a disconnected session can simply be re-run. Then copy `artifacts/cache_exp02/` back to the repository; the notebook
`02_Perception_Ensemble_Tiling_EXP02.ipynb` runs from the cache on CPU.

## EXP-02C (T4): calibrated multi-family ensemble + Submission #2

Everything runs from one notebook: **`notebooks/kaggle/T4_EXP02C_runner.ipynb`**. Upload it to Kaggle (from the
competition page, so the dataset is mounted) or open it in Colab. Set the accelerator to a **T4 GPU**, turn
**Internet** on, and *Run all*. The runner works through these steps:

1. Clones `feat/t4-ensemble-wbf` and runs the unit tests on the GPU box.
2. Runs a 2-image MM-Grounding-DINO smoke test, then the full resumable pass:
   `scripts/run_exp02_inference.py --backbone mmgdino_large` (`openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg`,
   one forward per semantic block and view).
3. Re-segments the EXP-01 boxes with SAM2.1 (for the paired Mask F1 comparison).
4. Executes `02_Perception_Ensemble_Tiling_EXP02.ipynb` end to end. Sections 3-5 fit per-detector/per-view
   calibration, anti-collision WBF and the sensitivity-group thresholds on CAL. They freeze
   `artifacts/cache_exp02/exp02c/config.json`, segment those boxes with `scripts/run_exp02c_segmentation.py` (SAM2.1
   Base+) and write `submissions/submission_exp02_perception.csv` (80 rows). Finally they run the 8-test suite with
   live inference, QA and VRAM hooks.
5. Packs `t4_outputs.zip`, containing the `mmgdino_large/` cache, `exp02c/`, the executed notebook and the submission.

Back on the workstation, unzip `t4_outputs.zip` at the repository root. `exp02c/scenes.jsonl` (~40 MB) is
git-ignored. `python scripts/build_notebook_02.py --refresh-markdown` inserts the written analyses into the
GPU-executed notebook without re-running it.

CPU code-path check without a GPU (OWLv2 Large stands in for MM-GDINO and masks are box fills; nothing it prints is a
result, and it writes only to the git-ignored `artifacts/cache_exp02/exp02c_dryrun/`):

```bash
TPI_EXP02C_DRY_RUN=1 python scripts/build_notebook_02.py --execute \
    --output artifacts/cache_exp02/exp02c_dryrun/nb02_dryrun.ipynb
```
