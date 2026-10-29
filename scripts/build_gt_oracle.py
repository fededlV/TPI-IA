"""T1 / S1 oracle: GT SceneJSON for the 20 QA images must validate and score Box F1 = Mask F1 = 1.0.

Usage: python3 scripts/build_gt_oracle.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config_and_seeds as cfg  # noqa: E402
from src import lvis_evaluator as ev  # noqa: E402
from src import scene_builder as sb  # noqa: E402


def build_gt_scenes(paths: cfg.DatasetPaths, annotations: dict, image_ids: list[int]) -> dict[str, dict]:
    categories = cfg.load_categories(paths.categories)
    images = {img["id"]: img for img in annotations["images"]}
    scenes = {}
    for image_id in image_ids:
        img = images[image_id]
        h, w = img["height"], img["width"]
        rgb = np.array(Image.open(paths.dev_images_dir / img["file_name"]).convert("RGB"))
        objects = []
        for ann in (a for a in annotations["annotations"] if a["image_id"] == image_id):
            mask = sb.polygon_to_mask(ann["segmentation"], h, w)
            objects.append(sb.build_object(f"gt_{ann['id']}", ann["category_id"], 1.0, sb.xywh_to_xyxy(ann["bbox"]),
                                           mask, image_rgb=rgb, category_name=categories[ann["category_id"]]))
        key = ev.image_key(img["file_name"])
        scenes[key] = sb.build_scene_json(key, w, h, objects)
    return scenes


def main() -> int:
    cfg.set_global_seeds(cfg.SEED)
    paths = cfg.get_dataset_paths()
    print(f"Dataset root: {paths.root}")
    annotations = json.loads(paths.annotations.read_text(encoding="utf-8"))
    schema = json.loads(paths.schema.read_text(encoding="utf-8"))
    qa_ids = cfg.load_qa_image_ids(paths.questions_dev)

    split = cfg.make_cal_val_split(annotations, qa_ids, seed=cfg.SEED)
    out = cfg.save_split(split)
    print(f"Split saved to {out}: CAL={len(split['cal'])} (QA {len(split['qa_cal'])}), "
          f"VAL={len(split['val'])} (QA {len(split['qa_val'])})")

    scenes = build_gt_scenes(paths, annotations, qa_ids)
    n_objects = sum(len(s["objects"]) for s in scenes.values())
    n_fallback = sum(1 for s in scenes.values() for o in s["objects"] if "mask_fallback" in o)
    schema_errors = {k: sb.validate_scene(s, schema) for k, s in scenes.items()}
    n_errors = sum(len(e) for e in schema_errors.values())
    print(f"GT SceneJSON: {len(scenes)} images, {n_objects} objects, "
          f"{sum(len(s['relations']) for s in scenes.values())} relations, "
          f"{n_fallback} bbox mask fallbacks, schema errors={n_errors}")

    report = ev.evaluate_federated({k: s["objects"] for k, s in scenes.items()}, ev.load_gt(annotations))
    print(ev.format_report(report))

    print("\nSuite on sample_submission.csv (require_qa=False):")
    print(ev.run_full_verification_suite(paths.sample_submission, require_qa=False))

    sample_rows = ev.read_submission(paths.sample_submission)
    with tempfile.TemporaryDirectory() as tmp:
        oracle_csv = Path(tmp) / "gt_oracle_submission.csv"
        with open(oracle_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "SceneJSON"])
            writer.writeheader()
            for row in sample_rows:
                scene = scenes.get(row["image_id"])
                writer.writerow({"image_id": row["image_id"],
                                 "SceneJSON": json.dumps(scene) if scene else row["SceneJSON"]})
        print("\nSuite on sample + 20 GT oracle rows (require_qa=False):")
        print(ev.run_full_verification_suite(oracle_csv, require_qa=False))

    ok = n_errors == 0 and report["box"]["f1"] == 1.0 and report["mask"]["f1"] == 1.0
    print(f"\nS1 ORACLE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
