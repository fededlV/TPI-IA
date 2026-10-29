"""EXP-01 baseline inference with caches (CPU-friendly, resumable).

  python scripts/run_exp01_inference.py detect              # OWLv2 Base on 120 dev + 60 test images
  python scripts/run_exp01_inference.py segment --threshold T  # SAM2.1 Base+ on boxes with score >= T

Inference reads only image files; annotations are never opened here.
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
from src.perception_detectors import OWLV2_BASE, PROMPT_TEMPLATE, OwlV2Detector, filter_detections  # noqa: E402
from src.segmentation_sam2 import SAM2_BASE_PLUS, Sam2Segmenter  # noqa: E402
from src.trace_logger import ExecutionTracer  # noqa: E402

CACHE_DIR = cfg.ARTIFACTS_DIR / "cache_exp01"
DETECTIONS_PATH = CACHE_DIR / "detections.json"


def scenes_path(threshold: float) -> Path:
    return CACHE_DIR / f"scenes_thr{threshold:.3f}.jsonl"


def image_files(paths: cfg.DatasetPaths) -> list[Path]:
    dev = sorted(paths.dev_images_dir.glob("tpi_*.jpg"))
    test = sorted(paths.test_images_dir.glob("test_*.jpg"))
    if len(dev) != 120 or len(test) != 60:
        raise RuntimeError(f"Expected 120 dev and 60 test images, found {len(dev)} and {len(test)}")
    return dev + test


def _environment() -> dict:
    import torch

    return {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "device": "cpu",
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "python": platform.python_version(),
            "machine": platform.processor() or platform.machine()}


def run_detect(paths: cfg.DatasetPaths) -> None:
    cfg.set_global_seeds(cfg.SEED)
    detector = OwlV2Detector(cfg.CATEGORIES)
    tracer = ExecutionTracer()
    cache = pl.load_detection_cache(DETECTIONS_PATH)[0] if DETECTIONS_PATH.exists() else {}
    meta = {"model": OWLV2_BASE, "prompt_template": PROMPT_TEMPLATE, "min_score": detector.min_score,
            "nms_iou": detector.nms_iou, **_environment()}
    files = image_files(paths)
    for n, f in enumerate(files, 1):
        if f.stem in cache:
            continue
        with tracer.stage(f.stem, "detection", {"model": OWLV2_BASE, "min_score": detector.min_score}) as rec:
            cache[f.stem] = detector.detect(Image.open(f))
            rec.set_counts(1, len(cache[f.stem]))
        pl.save_detection_cache(cache, DETECTIONS_PATH, meta)
        print(f"[{n}/{len(files)}] {f.stem}: {len(cache[f.stem])} raw detections", flush=True)
    tracer.to_jsonl(CACHE_DIR / "trace_detection.jsonl")


def run_segment(paths: cfg.DatasetPaths, threshold: float) -> None:
    cfg.set_global_seeds(cfg.SEED)
    detections, det_meta = pl.load_detection_cache(DETECTIONS_PATH)
    segmenter = Sam2Segmenter()
    tracer = ExecutionTracer()
    out = scenes_path(threshold)
    done = set()
    if out.exists():
        done = {json.loads(line)["image"]["id"] for line in out.read_text(encoding="utf-8").splitlines() if line}
    files = image_files(paths)
    with open(out, "a", encoding="utf-8") as f_out:
        for n, f in enumerate(files, 1):
            if f.stem in done:
                continue
            kept = filter_detections(detections[f.stem], threshold)
            scene, n_fallback = pl.segment_and_build(f.stem, Image.open(f).convert("RGB"), kept, segmenter,
                                                     cfg.CATEGORIES, tracer)
            f_out.write(json.dumps(scene) + "\n")
            f_out.flush()
            print(f"[{n}/{len(files)}] {f.stem}: {len(scene['objects'])} objects, {n_fallback} fallbacks", flush=True)
    meta = {"segmenter": SAM2_BASE_PLUS, "multimask_output": False, "score_threshold": threshold,
            "detections_meta": det_meta, **_environment()}
    (CACHE_DIR / f"scenes_thr{threshold:.3f}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tracer.to_jsonl(CACHE_DIR / f"trace_segmentation_thr{threshold:.3f}.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["detect", "segment"])
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args(argv)
    paths = cfg.get_dataset_paths()
    start = time.perf_counter()
    if args.stage == "detect":
        run_detect(paths)
    else:
        if args.threshold is None:
            parser.error("segment requires --threshold (tuned on CAL)")
        run_segment(paths, args.threshold)
    print(f"done in {time.perf_counter() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
