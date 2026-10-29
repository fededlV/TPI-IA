"""Build (and optionally execute) notebooks/01_EDA_and_Baseline_EXP01.ipynb from versionable cell sources.

  python scripts/build_notebook_01.py            # write the notebook only
  python scripts/build_notebook_01.py --execute  # write and execute it in place (outputs saved)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "01_EDA_and_Baseline_EXP01.ipynb"
ANALYSIS_DIR = REPO / "notebooks" / "analysis_01"


def md(name: str) -> str:
    """Analysis Markdown lives in notebooks/analysis_01/<name>.md (written from the executed outputs)."""
    return (ANALYSIS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


CELLS: list[tuple[str, str]] = [
    ("md", "00_title"),
    ("code", r'''
import csv, json, os, platform, sys, time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image

from src import config_and_seeds as cfg
from src import lvis_evaluator as ev
from src import pipeline as pl
from src import qa_engine as qa
from src import scene_builder as sb
from src.perception_detectors import filter_detections
from scripts.build_gt_oracle import build_gt_scenes

cfg.set_global_seeds(cfg.SEED)
pd.set_option("display.width", 160)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
PATHS = cfg.get_dataset_paths()
SCHEMA = json.loads(PATHS.schema.read_text(encoding="utf-8"))
ANNOTATIONS = json.loads(PATHS.annotations.read_text(encoding="utf-8"))  # EDA/evaluation only, never inference
QUESTIONS = [json.loads(l) for l in PATHS.questions_dev.read_text(encoding="utf-8").splitlines() if l.strip()]
CACHE_DIR = REPO / "artifacts" / "cache_exp01"
print(f"Python {platform.python_version()} | torch {torch.__version__} | transformers {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()} | CPU threads: {torch.get_num_threads()} | SEED={cfg.SEED}")
print(f"Dataset root: {PATHS.root}")
'''),
    ("md", "01_setup"),
    ("md", "10_section1"),
    ("code", r'''
obj_schema = SCHEMA["properties"]["objects"]["items"]
props = obj_schema["properties"]
rel_schema = SCHEMA["properties"]["relations"]
qa_def = SCHEMA["$defs"]["questionAnswer"]
audit = pd.DataFrame([
    ("root.required", SCHEMA["required"]),
    ("root optional (dev rows)", [k for k in SCHEMA["properties"] if k not in SCHEMA["required"]]),
    ("objects.maxItems", SCHEMA["properties"]["objects"]["maxItems"]),
    ("object.required (9)", obj_schema["required"]),
    ("bbox", json.dumps(props["bbox"])),
    ("area_ratio", json.dumps(props["area_ratio"])),
    ("position.enum", props["position"]["enum"]),
    ("size.enum", props["size"]["enum"]),
    ("relations.maxItems", rel_schema["maxItems"]),
    ("relation.predicate.enum", rel_schema["items"]["properties"]["predicate"]["enum"]),
    ("question.type.enum", qa_def["properties"]["type"]["enum"]),
    ("question.required", qa_def["required"]),
    ("normalized_answer", json.dumps(qa_def["properties"]["normalized_answer"])),
    ("questions_llm extra required", SCHEMA["$defs"]["questionAnswerLLM"]["allOf"][1]["required"]),
], columns=["contract", "value"])
print(audit.to_string(index=False, max_colwidth=110))
'''),
    ("md", "11_schema"),
    ("code", r'''
sample_rows = ev.read_submission(PATHS.sample_submission)
test_meta = pd.read_csv(PATHS.test_images)
ids = [r["image_id"] for r in sample_rows]
dev_ids = [i for i in ids if i.startswith("tpi_")]
qa_keys = sorted({q["image_file"].rsplit(".", 1)[0] for q in QUESTIONS})
print(f"rows={len(sample_rows)} | test={sum(i.startswith('test_') for i in ids)} | dev={len(dev_ids)}")
print("test order == test_images.csv:", ids[:60] == test_meta["image_id"].tolist())
print("dev rows == questions_dev images:", sorted(dev_ids) == qa_keys, "| dev order:", dev_ids)
example = json.loads(sample_rows[0]["SceneJSON"])
obj = example["objects"][0]
decoded = sb.decode_rle(obj["mask"])
print("example object:", {k: obj[k] for k in ("bbox", "centroid", "area_ratio", "position", "size")})
print("mask counts head:", obj["mask"]["counts"][:3], "| decoded pixels:", int(decoded.sum()),
      "| pixel (0,0) set:", bool(decoded[0, 0]), "| re-encoded equal:", sb.encode_rle(decoded) == obj["mask"])
print("dev row keys:", list(json.loads(sample_rows[-1]["SceneJSON"]).keys()))
sample_report = ev.run_full_verification_suite(PATHS.sample_submission, require_qa=False, raise_on_fail=False)
print(sample_report)
'''),
    ("md", "12_sample"),
    ("code", r'''
split = cfg.load_split()
regenerated = cfg.make_cal_val_split(ANNOTATIONS, cfg.load_qa_image_ids(PATHS.questions_dev), seed=cfg.SEED)
inst_per_image = Counter(a["image_id"] for a in ANNOTATIONS["annotations"])
split_df = pd.DataFrame([{
    "split": name, "images": len(split[name]), "qa_images": len(split[f"qa_{name}"]),
    "instances": sum(inst_per_image[i] for i in split[name]),
    "inst/img mean": np.mean([inst_per_image[i] for i in split[name]]).round(2),
    "inst/img median": float(np.median([inst_per_image[i] for i in split[name]])),
} for name in ("cal", "val")])
print(split_df.to_string(index=False))
print("disjoint:", not set(split["cal"]) & set(split["val"]), "| covers 1..120:",
      set(split["cal"]) | set(split["val"]) == set(range(1, 121)), "| reproducible:", regenerated == split)
print("tercile edges (instances/img):", split["tercile_edges"], "| QA CAL:", split["qa_cal"], "| QA VAL:", split["qa_val"])
'''),
    ("md", "13_split"),
    ("md", "20_section2"),
    ("code", r'''
img_by_id = {im["id"]: im for im in ANNOTATIONS["images"]}
ar_area = np.array([a["area"] / (img_by_id[a["image_id"]]["height"] * img_by_id[a["image_id"]]["width"])
                    for a in ANNOTATIONS["annotations"]])
ar_raster = np.array([sb.polygon_to_mask(a["segmentation"], img_by_id[a["image_id"]]["height"],
                                         img_by_id[a["image_id"]]["width"]).sum()
                      / (img_by_id[a["image_id"]]["height"] * img_by_id[a["image_id"]]["width"])
                      for a in ANNOTATIONS["annotations"]])
pcts = [10, 25, 33, 50, 66, 75, 90]
stats = pd.DataFrame({name: [len(ar), (ar < 0.01).mean(), *np.percentile(ar, pcts), (ar < cfg.SMALL_MAX_AREA_RATIO).mean(),
                             ((ar >= cfg.SMALL_MAX_AREA_RATIO) & (ar < cfg.MEDIUM_MAX_AREA_RATIO)).mean(),
                             (ar >= cfg.MEDIUM_MAX_AREA_RATIO).mean()]
                      for name, ar in (("annotation area / HW", ar_area), ("rasterized polygon / HW", ar_raster))},
                     index=["n", "share < 0.01", *[f"P{p}" for p in pcts], "small", "medium", "large"])
print(stats.round(5).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
bins = np.logspace(np.log10(ar_area.min()), 0, 50)
axes[0].hist(ar_area, bins=bins, color="#4C72B0")
axes[0].set_xscale("log"); axes[0].set_xlabel("area_ratio (log)"); axes[0].set_ylabel("instances")
axes[0].set_title("Plot 1a - GT instance scale (n=1,790)")
xs = np.sort(ar_area)
axes[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color="#4C72B0")
axes[1].set_xscale("log"); axes[1].set_xlabel("area_ratio (log)"); axes[1].set_ylabel("cumulative share")
axes[1].set_title("Plot 1b - CDF of area_ratio")
for ax in axes:
    for x, label, color in ((cfg.SMALL_MAX_AREA_RATIO, "small|medium 0.0035", "#DD8452"),
                            (cfg.MEDIUM_MAX_AREA_RATIO, "medium|large 0.035", "#55A868"), (0.01, "1% of image", "#C44E52")):
        ax.axvline(x, ls="--", color=color, label=label)
axes[1].axhline((ar_area < 0.01).mean(), color="#C44E52", lw=0.8)
axes[1].legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.show()
'''),
    ("md", "21_scale"),
    ("code", r'''
gt_cats = defaultdict(set)
for a in ANNOTATIONS["annotations"]:
    gt_cats[a["image_id"]].add(a["category_id"])
fed = pd.DataFrame([{"image_id": i, "positive": len(gt_cats[i]), "negative": len(im.get("neg_category_ids", [])),
                     "not_exhaustive": len(im.get("not_exhaustive_category_ids", []))}
                    for i, im in sorted(img_by_id.items())])
fed["unverified"] = 120 - fed["positive"] - fed["negative"]
print(fed[["positive", "negative", "not_exhaustive", "unverified"]].describe().round(2).to_string())
print(f"images with neg ids: {(fed.negative > 0).sum()} | with not_exhaustive ids: {(fed.not_exhaustive > 0).sum()}")
print(f"evaluable share of the vocabulary per image (pos+neg)/120: {((fed.positive + fed.negative) / 120).mean():.3f}")
neg_counter = Counter(c for im in ANNOTATIONS["images"] for c in im.get("neg_category_ids", []))
ne_counter = Counter(c for im in ANNOTATIONS["images"] for c in im.get("not_exhaustive_category_ids", []))
print("most frequent neg categories:", [(cfg.CATEGORIES[c], n) for c, n in neg_counter.most_common(8)])
print("most frequent not_exhaustive:", [(cfg.CATEGORIES[c], n) for c, n in ne_counter.most_common(8)])

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
order = fed.sort_values("positive", ascending=False).reset_index(drop=True)
axes[0].bar(order.index, order.positive, label="positive (FN risk)", color="#4C72B0")
axes[0].bar(order.index, order.negative, bottom=order.positive, label="neg_category_ids (FP risk)", color="#C44E52")
axes[0].bar(order.index, order.not_exhaustive, bottom=order.positive + order.negative,
            label="not_exhaustive (extra preds ignored)", color="#DD8452")
axes[0].set_xlabel("development images (sorted by #positive categories)"); axes[0].set_ylabel("categories")
axes[0].set_title("Plot 2a - Federated label state per image"); axes[0].legend(fontsize=8)
top_neg = neg_counter.most_common(12)
axes[1].barh([cfg.CATEGORIES[c] for c, _ in top_neg][::-1], [n for _, n in top_neg][::-1], color="#C44E52")
axes[1].set_xlabel("images where the category is verified absent"); axes[1].set_title("Plot 2b - Most frequent neg_category_ids")
plt.tight_layout(); plt.show()
'''),
    ("md", "22_federated"),
    ("code", r'''
counts = np.array([inst_per_image[i] for i in sorted(img_by_id)])
boxes_by = defaultdict(list)
for a in ANNOTATIONS["annotations"]:
    boxes_by[(a["image_id"], a["category_id"])].append(sb.xywh_to_xyxy(a["bbox"]))
intra = np.array([ev.box_iou(b[i], b[j]) for b in boxes_by.values() for i in range(len(b)) for j in range(i + 1, len(b))])
multi = Counter(cfg.CATEGORIES[c] for (_, c), b in boxes_by.items() if len(b) > 1)
print(f"instances/img: min={counts.min()} median={np.median(counts)} mean={counts.mean():.2f} max={counts.max()} | images >= 50: {(counts >= 50).sum()}")
print(f"same-class GT box pairs: {len(intra)} | IoU>0: {(intra > 0).mean():.3f} | IoU>=0.3: {(intra >= 0.3).mean():.4f} | "
      f"IoU>=0.5: {(intra >= 0.5).mean():.4f} ({(intra >= 0.5).sum()} pairs) | IoU>=0.55: {(intra >= 0.55).sum()} | max={intra.max():.3f}")
print("categories most often with >1 instance per image:", multi.most_common(8))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(counts, bins=range(0, counts.max() + 5, 3), color="#4C72B0")
axes[0].set_xlabel("GT instances per image"); axes[0].set_ylabel("images"); axes[0].set_title("Plot 3a - Scene density")
axes[1].hist(intra[intra > 0], bins=np.linspace(0, 1, 41), color="#8172B3")
axes[1].axvline(0.5, ls="--", color="#C44E52", label="NMS / match IoU 0.5")
axes[1].set_yscale("log"); axes[1].set_xlabel("IoU between same-class GT boxes (IoU>0 only)")
axes[1].set_ylabel("pairs (log)"); axes[1].set_title("Plot 3b - Intra-class overlap"); axes[1].legend()
plt.tight_layout(); plt.show()
'''),
    ("md", "23_density"),
    ("code", r'''
qa_ids = cfg.load_qa_image_ids(PATHS.questions_dev)
GT_SCENES = build_gt_scenes(PATHS, ANNOTATIONS, qa_ids)
q_df = pd.DataFrame(QUESTIONS)
q_df["image_key"] = q_df["image_file"].str.rsplit(".", n=1).str[0]
q_df["split"] = q_df["image_id"].map(lambda i: "CAL" if i in split["qa_cal"] else "VAL")
GT_ANSWERS = {q["question_id"]: qa.resolve_deterministic_qa(GT_SCENES[k], q, cfg.CATEGORIES)
              for q, k in zip(QUESTIONS, q_df["image_key"])}
q_df["gt_answer"] = q_df["question_id"].map(lambda i: GT_ANSWERS[i]["normalized_answer"])
q_df["n_evidence"] = q_df["question_id"].map(lambda i: len(GT_ANSWERS[i]["evidence"]))
print(pd.crosstab(q_df["type"], q_df["split"], margins=True).to_string())
closed = q_df[q_df.type != "description"]
print("\nGT-derived answers for the 40 closed questions:")
print(closed[["question_id", "type", "class_en", "gt_answer", "n_evidence"]].to_string(index=False))
print("\ncount answers are int:", all(type(a) is int for a in closed[closed.type == "count"].gt_answer))

def _gt_obj(key, name):
    cid = {v: k for k, v in cfg.CATEGORIES.items()}[name]
    return [o for o in GT_SCENES[key]["objects"] if o["category_id"] == cid]

size_rows = [(r.question_id, r.class_en[0], _gt_obj(r.image_key, r.class_en[0])[0]["area_ratio"], r.gt_answer)
             for r in q_df[q_df.type == "size"].itertuples()]
rel_rows = []
for r in q_df[q_df.type == "relation"].itertuples():
    a, b = _gt_obj(r.image_key, r.class_en[0])[0], _gt_obj(r.image_key, r.class_en[1])[0]
    w, h = GT_SCENES[r.image_key]["image"]["width"], GT_SCENES[r.image_key]["image"]["height"]
    d = np.hypot((a["centroid"][0] - b["centroid"][0]) / w, (a["centroid"][1] - b["centroid"][1]) / h)
    rel_rows.append((r.question_id, f"{r.class_en[0]} {qa.extract_relation_predicate(r.question)} {r.class_en[1]}", d, r.gt_answer))
print(pd.DataFrame(size_rows, columns=["question_id", "category", "area_ratio", "size"]).to_string(index=False))
print(pd.DataFrame(rel_rows, columns=["question_id", "relation", "norm. centroid dist", "answer"]).round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
q_df["type"].value_counts().plot.bar(ax=axes[0], color="#4C72B0"); axes[0].set_title("Plot 4a - Prompt types (60)")
yn = closed[closed.type.isin(["existence", "relation"])].groupby(["type", "gt_answer"]).size().unstack()
yn.plot.bar(ax=axes[1], color={"yes": "#55A868", "no": "#C44E52"}); axes[1].set_title("Plot 4b - yes/no balance")
axes[2].scatter([r[2] for r in size_rows], range(len(size_rows)), color="#8172B3")
for i, r in enumerate(size_rows):
    axes[2].annotate(f"{r[1]} ({r[3]})", (r[2], i), textcoords="offset points", xytext=(5, 3), fontsize=8)
for x in (cfg.SMALL_MAX_AREA_RATIO, cfg.MEDIUM_MAX_AREA_RATIO):
    axes[2].axvline(x, ls="--", color="gray")
axes[2].set_xscale("log"); axes[2].set_yticks([]); axes[2].set_xlabel("area_ratio (log)")
axes[2].set_title("Plot 4c - The 5 size questions vs thresholds")
plt.tight_layout(); plt.show()
'''),
    ("md", "24_qa"),
    ("md", "30_section3"),
    ("code", r'''
DETECTIONS, DET_META = pl.load_detection_cache(CACHE_DIR / "detections.json")
trace = [json.loads(l) for l in (CACHE_DIR / "trace_detection.jsonl").read_text().splitlines()]
det_ms = np.array([r["duration_ms"] for r in trace if r.get("type") == "stage"])
print("detection cache meta:", json.dumps(DET_META, indent=1))
print(f"images cached: {len(DETECTIONS)} | traced detection calls: {len(det_ms)} | "
      f"ms/img mean={det_ms.mean():.0f} median={np.median(det_ms):.0f} p95={np.percentile(det_ms, 95):.0f}")
raw_counts = np.array([len(v) for v in DETECTIONS.values()])
print(f"raw detections/img (score >= {DET_META['min_score']}): mean={raw_counts.mean():.1f} max={raw_counts.max()}")

# Reproducibility: fresh OWLv2 inference on one dev and one test image must reproduce the cache.
from src.perception_detectors import OwlV2Detector
DETECTOR = OwlV2Detector(cfg.CATEGORIES)
for key, folder in (("tpi_0003", PATHS.dev_images_dir), ("test_0007", PATHS.test_images_dir)):
    t0 = time.perf_counter()
    fresh = DETECTOR.detect(Image.open(folder / f"{key}.jpg"))
    dt = (time.perf_counter() - t0) * 1000
    cached = DETECTIONS[key]
    same_n = len(fresh) == len(cached)
    max_score = max((abs(a.score - b.score) for a, b in zip(fresh, cached)), default=0.0)
    max_box = max((max(abs(x - y) for x, y in zip(a.bbox, b.bbox)) for a, b in zip(fresh, cached)), default=0.0)
    print(f"{key}: fresh {len(fresh)} vs cached {len(cached)} | same labels: "
          f"{same_n and all(a.category_id == b.category_id for a, b in zip(fresh, cached))} | "
          f"max |dscore|={max_score:.2e} | max |dbox|={max_box:.2e}px | {dt:.0f} ms")
'''),
    ("md", "31_provenance"),
    ("code", r'''
GT = ev.load_gt(ANNOTATIONS)
key_of = {i: f"tpi_{i:04d}" for i in range(1, 121)}
CAL_KEYS = [key_of[i] for i in split["cal"]]
VAL_KEYS = [key_of[i] for i in split["val"]]

def as_preds(dets):
    return [{"category_id": d.category_id, "confidence": d.score, "bbox": list(d.bbox)} for d in dets]

def box_eval(keys, thr):
    return ev.evaluate_federated({k: as_preds(filter_detections(DETECTIONS[k], thr)) for k in keys}, GT, with_masks=False)

thresholds = np.round(np.arange(0.05, 0.501, 0.025), 3)
sweep = pd.DataFrame([{"threshold": t, **{f"CAL_{m}": box_eval(CAL_KEYS, t)["box"][m] for m in ("precision", "recall", "f1")}}
                      for t in thresholds])
best = sweep.loc[sweep["CAL_f1"].idxmax()]
FROZEN_THRESHOLD = float(best["threshold"])
val_at_frozen = box_eval(VAL_KEYS, FROZEN_THRESHOLD)["box"]
print(sweep.round(4).to_string(index=False))
print(f"\nFROZEN_THRESHOLD (argmax CAL Box F1) = {FROZEN_THRESHOLD} | CAL F1={best['CAL_f1']:.4f} | "
      f"VAL F1 at frozen threshold={val_at_frozen['f1']:.4f} (VAL not used for selection)")

fig, ax = plt.subplots(figsize=(8, 4))
for m, c in (("precision", "#4C72B0"), ("recall", "#DD8452"), ("f1", "#55A868")):
    ax.plot(sweep.threshold, sweep[f"CAL_{m}"], marker="o", ms=3, color=c, label=f"CAL Box {m}")
ax.axvline(FROZEN_THRESHOLD, ls="--", color="gray", label=f"frozen = {FROZEN_THRESHOLD}")
ax.axhline(0.45, ls=":", color="#C44E52", label="S2 target Box F1 0.45")
ax.set_xlabel("OWLv2 score threshold"); ax.set_ylabel("metric"); ax.set_title("Threshold sweep on CAL (80 images, box-only)")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
'''),
    ("md", "32_sweep"),
    ("code", r'''
scenes_file = CACHE_DIR / f"scenes_thr{FROZEN_THRESHOLD:.3f}.jsonl"
seg_meta = json.loads((CACHE_DIR / f"scenes_thr{FROZEN_THRESHOLD:.3f}.meta.json").read_text())
assert seg_meta["score_threshold"] == FROZEN_THRESHOLD, "scene cache was built with another threshold"
SCENES = {s["image"]["id"]: s for s in map(json.loads, scenes_file.read_text().splitlines())}
seg_trace = [json.loads(l) for l in (CACHE_DIR / f"trace_segmentation_thr{FROZEN_THRESHOLD:.3f}.jsonl").read_text().splitlines()]
stage_ms = defaultdict(dict)
for r in seg_trace:
    if r.get("type") == "stage":
        stage_ms[r["stage_name"]][r["image_id"]] = r["duration_ms"]
det_ms_by = {r["image_id"]: r["duration_ms"] for r in trace if r.get("type") == "stage"}
ON_GPU = seg_meta.get("device") == "cuda" and DET_META.get("device") == "cuda"
VRAM_PEAK_MB = max(r["vram_peak_mb"] for r in trace + seg_trace if r.get("type") == "stage") if ON_GPU else None
VRAM_LABEL = f"{VRAM_PEAK_MB:.0f} MB" if ON_GPU else "N/A (CPU)"
print("segmentation meta:", {k: seg_meta.get(k) for k in ("segmenter", "multimask_output", "score_threshold", "device", "created_at", "torch_threads")})
n_fallback = sum("mask_fallback" in o for s in SCENES.values() for o in s["objects"])
print(f"scenes: {len(SCENES)} | objects: {sum(len(s['objects']) for s in SCENES.values())} | bbox mask fallbacks: {n_fallback}")

def metrics_row(name, keys):
    r = ev.evaluate_federated({k: SCENES[k]["objects"] for k in keys}, GT)
    total = [det_ms_by[k] + stage_ms["segmentation"][k] + stage_ms["scene_build"][k] for k in keys]
    return {"split": name, "images": r["images"], "Box P": r["box"]["precision"], "Box R": r["box"]["recall"],
            "Box F1": r["box"]["f1"], "Mask P": r["mask"]["precision"], "Mask R": r["mask"]["recall"],
            "Mask F1": r["mask"]["f1"], "LVIS Coverage": r["coverage"],
            "objects/img": np.mean([len(SCENES[k]["objects"]) for k in keys]),
            "det ms/img": np.mean([det_ms_by[k] for k in keys]),
            "seg ms/img": np.mean([stage_ms["segmentation"][k] for k in keys]),
            "build ms/img": np.mean([stage_ms["scene_build"][k] for k in keys]),
            "total ms/img": np.mean(total), "VRAM peak": VRAM_LABEL}

TEST_KEYS = [f"test_{i:04d}" for i in range(1, 61)]
metrics = pd.DataFrame([metrics_row("CAL", CAL_KEYS), metrics_row("VAL", VAL_KEYS), metrics_row("DEV (120)", CAL_KEYS + VAL_KEYS)])
test_row = {"split": "TEST (60, no GT)", "images": 60, "objects/img": np.mean([len(SCENES[k]["objects"]) for k in TEST_KEYS]),
            "det ms/img": np.mean([det_ms_by[k] for k in TEST_KEYS]),
            "seg ms/img": np.mean([stage_ms["segmentation"][k] for k in TEST_KEYS]),
            "build ms/img": np.mean([stage_ms["scene_build"][k] for k in TEST_KEYS]), "VRAM peak": VRAM_LABEL}
test_row["total ms/img"] = test_row["det ms/img"] + test_row["seg ms/img"] + test_row["build ms/img"]
metrics = pd.concat([metrics, pd.DataFrame([test_row])], ignore_index=True)
print(metrics.round(4).to_string(index=False))
'''),
    ("md", "33_metrics"),
    ("code", r'''
def recall_by_bucket(keys):
    """Greedy one-to-one box matching per category (same rule as the evaluator) tagged by GT size bucket."""
    rows = []
    for k in keys:
        g = GT[k]
        h, w = g["height"], g["width"]
        for cid in g["pos"]:
            preds = sorted([o for o in SCENES[k]["objects"] if o["category_id"] == cid], key=lambda o: -o["confidence"])
            idx = [i for i, c in enumerate(g["categories"]) if c == cid]
            taken = set()
            for p in preds:
                cands = [(ev.box_iou(p["bbox"], g["boxes"][i]), i) for i in idx if i not in taken]
                best = max(cands, default=(0, None))
                if best[0] >= ev.IOU_THRESHOLD:
                    taken.add(best[1])
            for i in idx:
                ar = g["masks"][i].sum() / (h * w)
                rows.append({"image": k, "category": cfg.CATEGORIES[cid], "size": cfg.size_from_area_ratio(ar),
                             "matched": i in taken})
    return pd.DataFrame(rows)

rb = recall_by_bucket(CAL_KEYS + VAL_KEYS)
print("Box recall by GT size bucket (DEV 120):")
print(rb.groupby("size")["matched"].agg(["count", "mean"]).rename(columns={"mean": "recall"}).round(4).to_string())
fn_by_cat = rb[~rb.matched].groupby("category").size().sort_values(ascending=False)
print("\nTop-12 categories by false negatives:", fn_by_cat.head(12).to_dict())
fp_neg = Counter()
for k in CAL_KEYS + VAL_KEYS:
    for o in SCENES[k]["objects"]:
        if o["category_id"] in GT[k]["neg"]:
            fp_neg[cfg.CATEGORIES[o["category_id"]]] += 1
print("FP on verified-absent (neg) categories:", dict(fp_neg.most_common(10)))
'''),
    ("md", "34_errors"),
    ("code", r'''
per_image = sorted(((ev.evaluate_federated({k: SCENES[k]["objects"]}, GT, with_masks=False)["box"]["f1"], k) for k in VAL_KEYS))
picks = [per_image[-1], per_image[len(per_image) // 2], per_image[0]]
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, (f1, k) in zip(axes, picks):
    img = np.asarray(Image.open(PATHS.dev_images_dir / f"{k}.jpg").convert("RGB")).astype(float) / 255
    overlay = img.copy()
    for o in SCENES[k]["objects"]:
        m = sb.decode_rle(o["mask"])
        overlay[m] = 0.55 * overlay[m] + 0.45 * np.array([0.2, 0.5, 1.0])
    ax.imshow(overlay)
    for b in GT[k]["boxes"]:
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#55A868", lw=1.5))
    for o in SCENES[k]["objects"]:
        x1, y1, x2, y2 = o["bbox"]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#C44E52", lw=1, ls="--"))
        ax.text(x1, y1, f"{cfg.CATEGORIES[o['category_id']]} {o['confidence']:.2f}", fontsize=6, color="white",
                bbox=dict(facecolor="#C44E52", alpha=0.7, pad=0.5, lw=0))
    ax.set_title(f"{k} - Box F1 {f1:.2f} (green: GT, red: pred, blue: SAM2 mask)", fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()
'''),
    ("md", "35_qualitative"),
    ("code", r'''
qa_rows = []
for q in QUESTIONS:
    key = q["image_file"].rsplit(".", 1)[0]
    det = qa.resolve_deterministic_qa(SCENES[key], q, cfg.CATEGORIES)
    llm = qa.resolve_llm_mode_qa(SCENES[key], q["question_id"], q["question"], cfg.CATEGORIES)
    qa_rows.append({"question_id": q["question_id"], "type": q["type"],
                    "split": "CAL" if q["image_id"] in split["qa_cal"] else "VAL",
                    "gt_derived": GT_ANSWERS[q["question_id"]]["normalized_answer"],
                    "deterministic": det["normalized_answer"], "llm_mode": llm["normalized_answer"],
                    "evidence": len(det["evidence"])})
qa_pred = pd.DataFrame(qa_rows)
closed_pred = qa_pred[qa_pred.type != "description"].copy()
closed_pred["correct"] = [a == b and type(a) is type(b) for a, b in zip(closed_pred.deterministic, closed_pred.gt_derived)]
closed_pred["llm_agrees"] = closed_pred.deterministic == closed_pred.llm_mode
print(closed_pred.to_string(index=False))
print("\nDeterministic QA accuracy vs GT-derived answers (40 closed):")
print(closed_pred.groupby("type")["correct"].agg(["count", "mean"]).round(3).to_string())
print(closed_pred.groupby("split")["correct"].mean().round(3).to_dict(), "| overall:", round(closed_pred.correct.mean(), 3))
print("rule-based LLM-mode agreement with deterministic engine:", round(closed_pred.llm_agrees.mean(), 3))
print("\nExample description (tpi_0005, predicted scene):\n", qa.resolve_deterministic_qa(SCENES["tpi_0005"], QUESTIONS[15], cfg.CATEGORIES)["answer"])
'''),
    ("md", "36_qa"),
    ("md", "40_section4"),
    ("code", r'''
SUBMISSION = REPO / "submissions" / "submission_exp01_baseline.csv"
SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
questions_by_key = defaultdict(list)
for q in QUESTIONS:
    questions_by_key[q["image_file"].rsplit(".", 1)[0]].append(q)
with open(SUBMISSION, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "SceneJSON"])
    writer.writeheader()
    for row in sample_rows:
        key = row["image_id"]
        scene = dict(SCENES[key])
        if key in questions_by_key:
            scene["questions"] = qa.answer_questions(scene, questions_by_key[key], cfg.CATEGORIES)
            scene["questions_llm"] = qa.answer_questions_llm(scene, questions_by_key[key], cfg.CATEGORIES)
        writer.writerow({"image_id": key, "SceneJSON": json.dumps(scene, ensure_ascii=False)})
written = ev.read_submission(SUBMISSION)
first_dev = json.loads(written[60]["SceneJSON"])
print(f"{SUBMISSION.relative_to(REPO)}: {len(written)} rows, {SUBMISSION.stat().st_size / 1e6:.1f} MB")
print("dev row keys:", list(first_dev), "| questions:", len(first_dev["questions"]), "| questions_llm:", len(first_dev["questions_llm"]))
print("example questions_llm entry:", json.dumps(first_dev["questions_llm"][1], ensure_ascii=False, indent=1))
'''),
    ("md", "41_submission"),
    ("code", r'''
from src.segmentation_sam2 import Sam2Segmenter
SEGMENTER = Sam2Segmenter()

def infer_fn(image_path):
    scene, _ = pl.infer_scene(image_path, DETECTOR, SEGMENTER, cfg.CATEGORIES, FROZEN_THRESHOLD)
    return scene

t0 = time.perf_counter()
REPORT = ev.run_full_verification_suite(SUBMISSION, infer_fn=infer_fn, qa_fn=qa.answer_questions,
                                        vram_peak_mb=VRAM_PEAK_MB, raise_on_fail=False)
print(REPORT)
print(f"\nsuite time: {time.perf_counter() - t0:.1f}s | FAIL count: {len(REPORT.failed)}")
assert not REPORT.failed, "Submission #1 must pass the contract suite with 0 FAIL"
'''),
    ("md", "42_suite"),
    ("md", "43_conclusions"),
]


def build() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (TPI .venv)", "language": "python"}
    nb.cells = [new_markdown_cell(md(src)) if kind == "md" else new_code_cell(src.strip()) for kind, src in CELLS]
    return nb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    nb = build()
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    if args.execute:
        from nbclient import NotebookClient

        NotebookClient(nb, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    nbformat.write(nb, NOTEBOOK)
    print(f"wrote {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
