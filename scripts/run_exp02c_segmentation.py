"""EXP-02C scenes for Submission #2: frozen ensemble boxes -> SAM2.1 Base+ (multimask_output=False) -> SceneJSON.

  python scripts/run_exp02c_segmentation.py --config artifacts/cache_exp02/exp02c/config.json
  python scripts/run_exp02c_segmentation.py --config ... --box-masks   # CPU dry run: bbox-fill masks, no SAM2

The fused detections are rebuilt from the raw member caches with the same blind path as the notebook and the live
detector (detection_fusion.fuse_image). Output goes next to the config: scenes.jsonl (resumable; refuses to mix
scenes from another config), scenes.meta.json and trace_segmentation.jsonl. Annotations are never opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config_and_seeds as cfg  # noqa: E402
from src import detection_fusion as fu  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src.trace_logger import ExecutionTracer  # noqa: E402

SCENES_NAME = "scenes.jsonl"


class BoxFillSegmenter:
    """Dry-run stand-in for SAM2: every mask is its box (exercises the scene/CSV path on CPU, never reported)."""
    name = "box_fill"

    def segment(self, image, boxes) -> list[tuple[np.ndarray, float]]:
        w, h = image.size
        out = []
        for x1, y1, x2, y2 in boxes:
            m = np.zeros((h, w), dtype=bool)
            m[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = True
            out.append((m, 0.0))
        return out


def config_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def fused_detections(config: fu.EnsembleConfig, root) -> tuple[dict[str, list], dict[str, list[int]]]:
    """Final EXP-02C detections per image from the member caches (relative to `root`) and the image sizes."""
    raw, sizes = {}, {}
    for name, spec in config.members.items():
        dets, meta = pl.load_detection_cache(Path(root) / spec.cache)
        raw[name] = {k: fu.retag(v, name) for k, v in dets.items()}
        sizes.update(meta["image_sizes"])
    keys = sorted(set.intersection(*(set(r) for r in raw.values())))
    return {k: fu.fuse_image({n: raw[n][k] for n in raw}, *sizes[k], config) for k in keys}, sizes


def segment_scenes(dets_by_key: dict[str, list], image_paths, segmenter, out_dir, config_sha: str,
                   tracer: ExecutionTracer | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes_path, meta_path = out_dir / SCENES_NAME, out_dir / "scenes.meta.json"
    if scenes_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if (old.get("config_sha"), old.get("segmenter")) != (config_sha, segmenter.name):
            raise ValueError(f"{scenes_path} was built with config {old.get('config_sha')} / {old.get('segmenter')}, "
                             f"not {config_sha} / {segmenter.name}; delete it to re-segment")
    meta_path.write_text(json.dumps({"config_sha": config_sha, "segmenter": segmenter.name,
                                     "multimask_output": False}, indent=2), encoding="utf-8")
    done = set()
    if scenes_path.exists():
        done = {json.loads(line)["image"]["id"] for line in scenes_path.read_text(encoding="utf-8").splitlines() if line}
    tracer = tracer or ExecutionTracer()
    pending = [p for p in map(Path, image_paths) if p.stem not in done]
    with open(scenes_path, "a", encoding="utf-8") as f:
        for n, path in enumerate(pending, 1):
            scene, n_fallback = pl.segment_and_build(path.stem, Image.open(path).convert("RGB"), dets_by_key[path.stem],
                                                     segmenter, cfg.CATEGORIES, tracer)
            f.write(json.dumps(scene) + "\n")
            f.flush()
            print(f"[{n}/{len(pending)}] {path.stem}: {len(scene['objects'])} objects, {n_fallback} fallbacks",
                  flush=True)
    return scenes_path


def main(argv: list[str] | None = None) -> int:
    from scripts.run_exp02_inference import _environment, image_files

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--box-masks", action="store_true", help="dry run: bbox-fill masks instead of SAM2")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    cfg.set_global_seeds(cfg.SEED)
    config = fu.EnsembleConfig.load(args.config)
    dets, _ = fused_detections(config, cfg.REPO_ROOT)
    if args.box_masks:
        segmenter = BoxFillSegmenter()
    else:
        from src.segmentation_sam2 import Sam2Segmenter
        segmenter = Sam2Segmenter()
    tracer = ExecutionTracer()
    out = segment_scenes(dets, image_files(cfg.get_dataset_paths()), segmenter, args.config.parent,
                         config_sha(args.config), tracer)
    meta_path = args.config.parent / "scenes.meta.json"
    meta = {**json.loads(meta_path.read_text(encoding="utf-8")), "segmenter_model": getattr(segmenter, "model_id", None),
            **_environment()}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with open(args.config.parent / "trace_segmentation.jsonl", "a", encoding="utf-8") as f:
        for rec in tracer.to_records():
            f.write(json.dumps({"type": "stage", **rec}) + "\n")
    print(f"wrote {out} in {time.perf_counter() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
