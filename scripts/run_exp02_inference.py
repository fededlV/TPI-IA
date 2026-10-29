"""EXP-02 raw detection caches (GPU recommended: Kaggle/Colab T4; resumable).

  python scripts/run_exp02_inference.py --backbone base    # OWLv2 Base  (controlled link to EXP-01)
  python scripts/run_exp02_inference.py --backbone large   # OWLv2 Large
  python scripts/run_exp02_inference.py --backbone mmgdino_large [--prompts alias]   # EXP-02C second family
  python scripts/run_exp02_inference.py --backbone base --limit 2   # smoke test

Per image: one global pass + four 2x2 tile passes (25% overlap). OWLv2: a single forward per view over the official
names plus LVIS_PROMPT_ALIASES, from which the four EXP-02A variants (single/chunked x raw/alias) are extracted.
(MM-)Grounding-DINO: one forward per semantic block and view (256-token text budget). Tile boxes are
cached projected to the global frame and NOT truncation-filtered (merge_views does that offline, so the filter and
the tiling category subset can be ablated without GPU). Inference reads only image files; annotations are never
opened here.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config_and_seeds as cfg  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src.perception_detectors import (GDINO_BASE, MM_GDINO_LARGE, OWLV2_BASE, OWLV2_LARGE,  # noqa: E402
                                      TILE_OVERLAP, GroundingDinoDetector, OwlV2Detector, detect_with_tiles)
from src.trace_logger import ExecutionTracer  # noqa: E402

CACHE_DIR = cfg.ARTIFACTS_DIR / "cache_exp02"
MODELS = {"base": OWLV2_BASE, "large": OWLV2_LARGE, "mmgdino_large": MM_GDINO_LARGE, "gdino_base": GDINO_BASE}
CACHE_NDIGITS = 4
SAVE_EVERY = 10


def detections_path(backbone: str, variant: str) -> Path:
    return CACHE_DIR / backbone / f"detections_{variant}.json.gz"


def image_files(paths: cfg.DatasetPaths) -> list[Path]:
    dev = sorted(paths.dev_images_dir.glob("tpi_*.jpg"))
    test = sorted(paths.test_images_dir.glob("test_*.jpg"))
    if len(dev) != 120 or len(test) != 60:
        raise RuntimeError(f"Expected 120 dev and 60 test images, found {len(dev)} and {len(test)}")
    return dev + test


def _environment() -> dict:
    import torch

    device = cfg.resolve_device()
    return {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "python": platform.python_version(),
            "machine": platform.processor() or platform.machine()}


def make_detector(backbone: str, prompts: str):
    if backbone in ("base", "large"):
        if prompts != "raw":
            raise ValueError("OWLv2 always caches the raw and alias variants from one pass; omit --prompts")
        return OwlV2Detector(cfg.CATEGORIES, model_id=MODELS[backbone])
    return GroundingDinoDetector(cfg.CATEGORIES, model_id=MODELS[backbone], use_aliases=prompts == "alias")


def run(paths: cfg.DatasetPaths, backbone: str, limit: int | None, prompts: str = "raw") -> None:
    cfg.set_global_seeds(cfg.SEED)
    detector = make_detector(backbone, prompts)
    variants = detector.variants
    tracer = ExecutionTracer()
    caches = {v: pl.load_detection_cache(detections_path(backbone, v))[0] if detections_path(backbone, v).exists()
              else {} for v in variants}
    prev_meta = (pl.load_detection_cache(detections_path(backbone, variants[0]))[1]
                 if detections_path(backbone, variants[0]).exists() else {})
    image_sizes = dict(prev_meta.get("image_sizes", {}))
    meta = {**detector.cache_meta(), "tile_overlap": TILE_OVERLAP, "aliases": cfg.LVIS_PROMPT_ALIASES,
            "views": ["global", "tile0..tile3"], "truncation_filtered": False, **_environment()}

    def save() -> None:
        for v in variants:
            pl.save_detection_cache(caches[v], detections_path(backbone, v), {**meta, "variant": v,
                                    "image_sizes": image_sizes}, ndigits=CACHE_NDIGITS)

    files = image_files(paths)[:limit]
    pending = [f for f in files if not all(f.stem in caches[v] for v in variants)]
    for n, f in enumerate(pending, 1):
        image = Image.open(f).convert("RGB")
        with tracer.stage(f.stem, "detection_exp02", {"model": MODELS[backbone], "views": 5}) as rec:
            out = detect_with_tiles(image, detector.detect_variants, base_source=detector.source)
            rec.set_counts(5, sum(len(d) for d in out.values()))
        for v in variants:
            caches[v][f.stem] = out[v]
        image_sizes[f.stem] = list(image.size)
        if n % SAVE_EVERY == 0 or n == len(pending):
            save()
        print(f"[{n}/{len(pending)}] {f.stem}: " + " ".join(f"{v}={len(out[v])}" for v in variants)
              + f" ({rec.duration_ms / 1000:.1f}s)", flush=True)
    # OWLv2 caches all variants from one pass; a GDINO prompt variant is its own pass with its own trace
    trace_name = "trace_detection.jsonl" if len(variants) > 1 else f"trace_detection_{variants[0]}.jsonl"
    trace_path = CACHE_DIR / backbone / trace_name
    if tracer.to_records():
        with open(trace_path, "a", encoding="utf-8") as f:
            for rec in tracer.to_records():
                f.write(json.dumps({"type": "stage", **rec}) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=sorted(MODELS), required=True)
    parser.add_argument("--prompts", choices=["raw", "alias"], default="raw",
                        help="GDINO family only: official names or official names + LVIS aliases")
    parser.add_argument("--limit", type=int, help="only the first N images (smoke test)")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    run(cfg.get_dataset_paths(), args.backbone, args.limit, args.prompts)
    print(f"done in {time.perf_counter() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
