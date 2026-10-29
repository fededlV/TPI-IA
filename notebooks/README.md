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
