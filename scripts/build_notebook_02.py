"""Build (and optionally execute) notebooks/02_Perception_Ensemble_Tiling_EXP02.ipynb.

  python scripts/build_notebook_02.py                     # write the notebook only
  python scripts/build_notebook_02.py --execute           # write and execute it (outputs saved)
  python scripts/build_notebook_02.py --refresh-markdown  # re-insert analysis Markdown into the executed notebook
  TPI_EXP02C_DRY_RUN=1 python scripts/build_notebook_02.py --execute --output <path>   # CPU code-path check

Sections 1-2 (EXP-02A/B) run from the OWLv2 GPU caches on CPU. Sections 3-5 (EXP-02C, scripts/notebook_02_exp02c.py)
need the MM-GDINO cache, and segment the frozen EXP-02C boxes with SAM2.1 on a GPU unless the scenes are cached.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

sys.path.insert(0, str(Path(__file__).resolve().parent))

from notebook_02_exp02c import EXP02C_CELLS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "02_Perception_Ensemble_Tiling_EXP02.ipynb"
ANALYSIS_DIR = REPO / "notebooks" / "analysis_02"


def md(name: str) -> str:
    """Analysis Markdown lives in notebooks/analysis_02/<name>.md (written from the executed outputs)."""
    path = ANALYSIS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8").strip() if path.exists() else f"*(analysis `{name}` pending)*"


CELLS: list[tuple[str, str]] = [
    ("md", "00_title"),
    ("code", r'''
import json, platform, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image

from src import config_and_seeds as cfg
from src import lvis_evaluator as ev
from src import pipeline as pl
from src import perception_detectors as pdet

cfg.set_global_seeds(cfg.SEED)
pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 140)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
PATHS = cfg.get_dataset_paths()
ANNOTATIONS = json.loads(PATHS.annotations.read_text(encoding="utf-8"))  # evaluation only, never inference
GT = ev.load_gt(ANNOTATIONS)
SPLIT = cfg.load_split()
CAL_KEYS = [f"tpi_{i:04d}" for i in SPLIT["cal"]]
VAL_KEYS = [f"tpi_{i:04d}" for i in SPLIT["val"]]  # held out: only Sections 3-5 report it, never used for selection
DEV_KEYS = CAL_KEYS + VAL_KEYS
AREA_RATIO = {k: [float(m.sum()) / (g["height"] * g["width"]) for m in g["masks"]] for k, g in GT.items()}

CACHE_DIR = REPO / "artifacts" / "cache_exp02"
BACKBONES = ("base", "large")
DETS, META = {}, {}
for bb in BACKBONES:
    for v in pdet.VARIANTS:
        DETS[bb, v], META[bb, v] = pl.load_detection_cache(CACHE_DIR / bb / f"detections_{v}.json.gz")
SIZES = META["base", "single_raw"]["image_sizes"]  # [W, H] per image, recorded by the runner
EXP01 = pl.load_detection_cache(REPO / "artifacts" / "cache_exp01" / "detections.json")[0]
TRACE = {bb: [json.loads(l) for l in (CACHE_DIR / bb / "trace_detection.jsonl").read_text().splitlines() if l.strip()]
         for bb in BACKBONES}

print(f"Python {platform.python_version()} | SEED={cfg.SEED} | CAL images: {len(CAL_KEYS)} | "
      f"CAL GT instances: {sum(len(GT[k]['boxes']) for k in CAL_KEYS)}")
display(pd.DataFrame([{"backbone": bb, "model": m["model"], "device": m["device"], "gpu": m["gpu"], "torch": m["torch"],
                       "created_at": m["created_at"], "text queries": m["n_queries"],
                       "images": len(DETS[bb, "single_raw"]), "tile overlap": m["tile_overlap"],
                       "score floor": m["min_score"], "truncation-filtered in cache": m["truncation_filtered"]}
                      for bb in BACKBONES for m in [META[bb, "single_raw"]]]))
'''),
    ("code", r'''
# Shared evaluation helpers: box-only LVIS federated matching (IoU >= 0.5), per image, so that every comparison can be
# bootstrapped paired by image. Size buckets use the rasterized GT mask area_ratio (same definition as EXP-01).
THRESHOLDS = np.round(np.arange(0.05, 0.601, 0.025), 3)
TILE_THRESHOLDS = np.round(np.arange(0.15, 0.501, 0.025), 3)
BUCKETS = ("small", "medium", "large")
GT_BUCKET = {k: [cfg.size_from_area_ratio(a) for a in AREA_RATIO[k]] for k in DEV_KEYS}
GROUPS = pdet.sibling_groups(cfg.CATEGORIES)
METRICS = ["P", "R", "F1", "R_small", "R_medium", "R_large"]


def as_preds(dets):
    return [{"category_id": d.category_id, "confidence": d.score, "bbox": list(d.bbox)} for d in dets]


def is_global(d):
    return pdet.view_index(d.source) is None


def global_only(dets_by_key, keys=None):
    return {k: [d for d in dets_by_key[k] if is_global(d)] for k in (keys or CAL_KEYS)}


def at_threshold(dets_by_key, thr, keys=None):
    return {k: [d for d in dets_by_key[k] if d.score >= thr] for k in (keys or CAL_KEYS)}


def per_image_counts(dets_by_key, keys=None):
    """[n_images, 9]: tp, fp, fn, then (matched, total) GT instances for small / medium / large."""
    keys = keys or CAL_KEYS
    per = ev.box_match_details({k: as_preds(dets_by_key[k]) for k in keys}, GT)
    rows = []
    for k in keys:
        row = [per[k]["tp"], per[k]["fp"], per[k]["fn"]]
        for b in BUCKETS:
            flags = [hit for hit, s in zip(per[k]["gt_matched"], GT_BUCKET[k]) if s == b]
            row += [sum(flags), len(flags)]
        rows.append(row)
    return np.array(rows, dtype=float)


def metrics_from_sums(s):
    """Vectorised over leading axes: s[..., 9] summed counts -> dict of metric arrays."""
    tp, fp, fn = s[..., 0], s[..., 1], s[..., 2]
    p, r = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1)
    out = {"P": p, "R": r, "F1": np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)}
    for i, b in enumerate(BUCKETS):
        out[f"R_{b}"] = s[..., 3 + 2 * i] / np.maximum(s[..., 4 + 2 * i], 1)
    return out


def summarize(dets_by_key, name, keys=None, **extra):
    keys = keys or CAL_KEYS
    counts = per_image_counts(dets_by_key, keys)
    m = metrics_from_sums(counts.sum(axis=0))
    return {"arm": name, **extra, **{k: float(m[k]) for k in METRICS},
            "det/img": float(np.mean([len(dets_by_key[k]) for k in keys]))}


def sweep(make, thresholds, keys=None):
    return pd.DataFrame([{"threshold": float(t), **summarize(make(t), str(t), keys)} for t in thresholds])


def bootstrap_delta(counts_a, counts_b, n_boot=1000, seed=cfg.SEED):
    """Paired image-level bootstrap of (b - a) per metric: point estimate, 95% percentile CI, P(delta <= 0)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts_a), size=(n_boot, len(counts_a)))
    ma, mb = metrics_from_sums(counts_a[idx].sum(axis=1)), metrics_from_sums(counts_b[idx].sum(axis=1))
    pa, pb = metrics_from_sums(counts_a.sum(axis=0)), metrics_from_sums(counts_b.sum(axis=0))
    rows = {}
    for k in METRICS:
        d = mb[k] - ma[k]
        rows[k] = {"delta": float(pb[k] - pa[k]), "ci_low": float(np.percentile(d, 2.5)),
                   "ci_high": float(np.percentile(d, 97.5)), "p(delta<=0)": float(np.mean(d <= 0))}
    return rows


n_bucket = Counter(b for k in CAL_KEYS for b in GT_BUCKET[k])
print("CAL GT instances by size bucket:", dict(n_bucket),
      f"| small share = {n_bucket['small'] / sum(n_bucket.values()):.1%}")
'''),
    ("md", "01_setup"),
    ("md", "10_section1"),
    ("code", r'''
BLOCKS = pdet.build_blocks(cfg.CATEGORIES)
QUERY_TABLE = pdet.build_query_table(cfg.LVIS_PROMPT_ALIASES)
display(pd.DataFrame([{"block": name, "classes": len(ids), "members": ", ".join(cfg.CATEGORIES[c] for c in ids)}
                      for name, ids in BLOCKS.items()]))
alias_rows = [{"id": cid, "official name": e["name"], "visual aliases (remapped to the official id)": ", ".join(e["prompts"][1:])}
              for cid, e in cfg.LVIS_PROMPT_ALIASES.items() if len(e["prompts"]) > 1]
print(f"{len(alias_rows)} of 120 categories carry aliases -> {len(QUERY_TABLE.prompts)} text queries "
      f"(template {pdet.PROMPT_TEMPLATE!r}); every query maps to exactly one category_id")
display(pd.DataFrame(alias_rows))
display(pd.DataFrame([{"sibling group": " / ".join(cfg.CATEGORIES[c] for c in sorted(g)),
                       "block": next(n for n, ids in BLOCKS.items() if g <= set(ids))} for g in GROUPS]))
'''),
    ("md", "11_prompt_design"),
    ("code", r'''
# Regression: the single-block raw variant of the alias-union pass (T4 GPU) must reproduce the EXP-01 cache (CPU).
ref, new = {k: EXP01[k] for k in CAL_KEYS}, global_only(DETS["base", "single_raw"])
same_count = sum(len(ref[k]) == len(new[k]) for k in CAL_KEYS)
key_of = lambda d: (d.category_id, tuple(round(v, 1) for v in d.bbox))
diffs, unmatched = [], 0
for k in CAL_KEYS:
    by_key = {key_of(d): d.score for d in new[k]}
    for d in ref[k]:
        if key_of(d) in by_key:
            diffs.append(abs(by_key[key_of(d)] - d.score))
        else:
            unmatched += 1
print(f"images with identical detection counts: {same_count}/{len(CAL_KEYS)} | EXP-01 detections without a "
      f"counterpart: {unmatched} | max |score diff| = {max(diffs):.1e} (cache rounding 1e-4 + GPU/CPU float)")
display(pd.DataFrame([summarize(at_threshold(ref, 0.175), "EXP-01 cache (CPU, 120-query pass)"),
                      summarize(at_threshold(new, 0.175), "EXP-02 base single_raw, global view (T4 GPU, 183-query pass)")]
                     ).round(4))
'''),
    ("md", "12_regression"),
    ("code", r'''
rows, CURVES = [], {}
for bb in BACKBONES:
    for v in pdet.VARIANTS:
        g = global_only(DETS[bb, v])
        sw = sweep(lambda t: at_threshold(g, t), THRESHOLDS)
        CURVES[bb, v] = sw
        best = sw.loc[sw["F1"].idxmax()]
        rows.append({**summarize(at_threshold(g, best["threshold"]), f"{bb} / {v}"), "backbone": bb, "variant": v,
                     "threshold": best["threshold"], "R@0.05 (ceiling)": sw.iloc[0]["R"]})
EXP02A = pd.DataFrame(rows)
BEST_T = {(r.backbone, r.variant): r.threshold for r in EXP02A.itertuples()}
display(EXP02A[["arm", "threshold", "P", "R", "F1", "R_small", "R_medium", "R_large", "R@0.05 (ceiling)", "det/img"]]
        .round(4))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, bb in zip(axes, BACKBONES):
    for v in pdet.VARIANTS:
        ax.plot(CURVES[bb, v]["threshold"], CURVES[bb, v]["F1"], marker=".", label=v)
    ax.set(title=f"OWLv2 {bb}: CAL Box F1 vs threshold (global view)", xlabel="score threshold", ylabel="Box F1")
    ax.legend()
plt.tight_layout()
plt.show()
'''),
    ("md", "13_factorial"),
    ("code", r'''
COUNTS_A = {key: per_image_counts(at_threshold(global_only(DETS[key]), t)) for key, t in BEST_T.items()}
rows = []
for bb in BACKBONES:
    for v in pdet.VARIANTS[1:]:
        boot = bootstrap_delta(COUNTS_A[bb, "single_raw"], COUNTS_A[bb, v])
        rows += [{"comparison": f"{bb}: {v} - single_raw", "metric": m, **boot[m]} for m in ("F1", "P", "R", "R_small")]
boot = bootstrap_delta(COUNTS_A["base", "single_raw"], COUNTS_A["large", "single_raw"])
rows += [{"comparison": "single_raw: large - base", "metric": m, **boot[m]} for m in ("F1", "P", "R", "R_small")]
BOOT_A = pd.DataFrame(rows)
BOOT_A["significant (95% CI excludes 0)"] = (BOOT_A["ci_low"] > 0) | (BOOT_A["ci_high"] < 0)
display(BOOT_A.round(4))
'''),
    ("md", "14_bootstrap"),
    ("code", r'''
def recall_by_category(dets_by_key):
    per = ev.box_match_details({k: as_preds(dets_by_key[k]) for k in CAL_KEYS}, GT)
    hits = defaultdict(lambda: [0, 0])
    for k in CAL_KEYS:
        for cid, hit in zip(GT[k]["categories"], per[k]["gt_matched"]):
            hits[cid][0] += hit
            hits[cid][1] += 1
    return hits

def n_pred_by_category(dets_by_key):
    return Counter(d.category_id for k in CAL_KEYS for d in dets_by_key[k])

raw = at_threshold(global_only(DETS["base", "single_raw"]), BEST_T["base", "single_raw"])
alias = at_threshold(global_only(DETS["base", "single_alias"]), BEST_T["base", "single_alias"])
h_raw, h_alias = recall_by_category(raw), recall_by_category(alias)
p_raw, p_alias = n_pred_by_category(raw), n_pred_by_category(alias)
rows = [{"category": e["name"], "aliases": ", ".join(e["prompts"][1:]), "CAL GT": h_raw[cid][1],
         "recall raw": h_raw[cid][0] / h_raw[cid][1], "recall alias": h_alias[cid][0] / h_alias[cid][1],
         "predictions raw": p_raw[cid], "predictions alias": p_alias[cid]}
        for cid, e in cfg.LVIS_PROMPT_ALIASES.items() if len(e["prompts"]) > 1 and h_raw[cid][1] > 0]
ALIAS_CAT = pd.DataFrame(rows)
ALIAS_CAT["delta recall"] = ALIAS_CAT["recall alias"] - ALIAS_CAT["recall raw"]
display(ALIAS_CAT.sort_values(["delta recall", "CAL GT"], ascending=[False, False]).round(3))
tot = ALIAS_CAT[["CAL GT", "predictions raw", "predictions alias"]].sum()
gained = int(round((ALIAS_CAT["delta recall"] * ALIAS_CAT["CAL GT"]).sum()))
print(f"aliased categories present in CAL: {len(ALIAS_CAT)} | GT instances: {int(tot['CAL GT'])} | net matched "
      f"instances gained: {gained:+d} | predictions {int(tot['predictions raw'])} -> {int(tot['predictions alias'])} "
      f"(base, single block, each at its CAL-optimal threshold)")
'''),
    ("md", "15_aliases"),
    ("code", r'''
def sibling_confusions(dets_by_key):
    """Missed GT instances of a sibling class X that a look-alike sibling Y localises (IoU >= 0.5)."""
    per = ev.box_match_details({k: as_preds(dets_by_key[k]) for k in CAL_KEYS}, GT)
    n_gt = n_conf = 0
    for k in CAL_KEYS:
        for cid, box, hit in zip(GT[k]["categories"], GT[k]["boxes"], per[k]["gt_matched"]):
            group = next((g for g in GROUPS if cid in g), None)
            if group is None:
                continue
            n_gt += 1
            n_conf += (not hit) and any(d.category_id in group and d.category_id != cid
                                        and ev.box_iou(list(d.bbox), box) >= ev.IOU_THRESHOLD for d in dets_by_key[k])
    return n_gt, n_conf

rows = []
for bb in BACKBONES:
    for v in pdet.VARIANTS:
        g = global_only(DETS[bb, v])
        op = at_threshold(g, BEST_T[bb, v])
        sup_op = {k: pdet.suppress_siblings(op[k], GROUPS) for k in CAL_KEYS}
        sup_floor = sum(len(g[k]) - len(pdet.suppress_siblings(g[k], GROUPS)) for k in CAL_KEYS)
        n_gt, n_conf = sibling_confusions(op)
        rows.append({"arm": f"{bb} / {v}", "suppressed @ operating thr": sum(len(op[k]) - len(sup_op[k]) for k in CAL_KEYS),
                     "suppressed @ 0.05 floor": sup_floor, "F1 without rule": summarize(op, "")["F1"],
                     "F1 with rule": summarize(sup_op, "")["F1"], "sibling GT in CAL": n_gt,
                     "missed GT localised by a sibling label": n_conf})
display(pd.DataFrame(rows).round(4))
'''),
    ("md", "16_siblings"),
    ("code", r'''
WIN = EXP02A.loc[EXP02A["F1"].idxmax()]
WIN_KEY, WIN_T = (WIN["backbone"], WIN["variant"]), float(WIN["threshold"])
print(f"EXP-02A selection rule (max CAL Box F1 over the 8 global arms): backbone={WIN_KEY[0]}, variant={WIN_KEY[1]}, "
      f"threshold={WIN_T}, F1={WIN['F1']:.4f}, R_small={WIN['R_small']:.4f}")
'''),
    ("md", "17_decision_02a"),
    ("md", "20_section2"),
    ("code", r'''
n_small = {k: sum(b == "small" for b in GT_BUCKET[k]) for k in CAL_KEYS}
DEMO = max(CAL_KEYS, key=lambda k: (n_small[k], k))
W, H = SIZES[DEMO]
tiles = pdet.tile_grid(W, H)
tw, th = tiles[0][2], tiles[0][3]
print(f"{DEMO}: {W}x{H} px | tile {tw}x{th} px | overlap band {2 * tw - W} px horizontally "
      f"({(2 * tw - W) / tw:.1%} of the tile width), {2 * th - H} px vertically ({(2 * th - H) / th:.1%}) | "
      f"inner-edge margin {pdet.TILE_EDGE_MARGIN:.0f} px | small GT instances: {n_small[DEMO]}")
fig, ax = plt.subplots(figsize=(7, 7 * H / W))
ax.imshow(Image.open(PATHS.dev_images_dir / f"{DEMO}.jpg"))
for i, (x1, y1, x2, y2) in enumerate(tiles):
    ax.add_patch(mpatches.Rectangle((x1 + 2 * i, y1 + 2 * i), x2 - x1 - 4 * i, y2 - y1 - 4 * i, fill=False,
                                    lw=2, ls="--", ec=f"C{i}", label=f"tile{i} {tiles[i]}"))
ax.axvspan(tiles[1][0], tiles[0][2], color="white", alpha=0.15)
ax.axhspan(tiles[2][1], tiles[0][3], color="white", alpha=0.15)
ax.set(title=f"2x2 tiles with 25% overlap on {DEMO} (shaded = overlap bands)", xticks=[], yticks=[])
ax.grid(False)
ax.legend(loc="lower left", fontsize=7)
plt.show()
'''),
    ("md", "21_geometry"),
    ("code", r'''
by_cat = defaultdict(list)
for k in CAL_KEYS:
    for cid, ar in zip(GT[k]["categories"], AREA_RATIO[k]):
        by_cat[cid].append(ar)
TILE_CATEGORIES = sorted(c for c in cfg.CATEGORIES
                         if c not in by_cat or np.median(by_cat[c]) < cfg.MEDIUM_MAX_AREA_RATIO)
excluded = sorted((c for c in cfg.CATEGORIES if c not in TILE_CATEGORIES), key=lambda c: -np.median(by_cat[c]))
display(pd.DataFrame([{"category": cfg.CATEGORIES[c], "CAL GT": len(by_cat[c]),
                       "median area_ratio": float(np.median(by_cat[c]))} for c in excluded]).round(4))
print(f"tiled categories: {len(TILE_CATEGORIES)}/120 (CAL GT median area_ratio < {cfg.MEDIUM_MAX_AREA_RATIO}; "
      f"{sum(c not in by_cat for c in cfg.CATEGORIES)} absent from CAL, included) | excluded (large): {len(excluded)}")
TILING_PATH = CACHE_DIR / "tiling_categories.json"
TILING_PATH.write_text(json.dumps({
    "rule": f"CAL GT median area_ratio < {cfg.MEDIUM_MAX_AREA_RATIO} (categories absent from CAL included)",
    "split": "CAL", "category_ids": TILE_CATEGORIES,
    "excluded": {cfg.CATEGORIES[c]: round(float(np.median(by_cat[c])), 5) for c in excluded}}, indent=2),
    encoding="utf-8")
print(f"wrote {TILING_PATH.relative_to(REPO)}")
'''),
    ("md", "22_subset"),
    ("code", r'''
ALL_CATEGORIES = set(cfg.CATEGORIES)


def tiled(dets_by_key, global_thr, tile_thr, categories, gate, drop_truncated=True, keys=None):
    """Global view at its own threshold + tile views at theirs, merged by merge_views (filter, gate, NMS)."""
    return {k: pdet.merge_views([d for d in dets_by_key[k] if not is_global(d) or d.score >= global_thr],
                                *SIZES[k], set(categories), drop_truncated=drop_truncated, tile_min_score=tile_thr,
                                tile_max_area_ratio=cfg.MEDIUM_MAX_AREA_RATIO if gate else None)
            for k in (keys or CAL_KEYS)}


def best_of(make, thresholds):
    sw = sweep(make, thresholds)
    return float(sw.loc[sw["F1"].idxmax(), "threshold"])


D = DETS[WIN_KEY]
G = global_only(D)
ARMS = {}
ARMS["B0 full image"] = (WIN_T, None, at_threshold(G, WIN_T))
t = best_of(lambda t: tiled(D, t, t, ALL_CATEGORIES, False, drop_truncated=False), THRESHOLDS)
ARMS["B1 + tiles, one shared threshold, no truncation filter"] = (t, t, tiled(D, t, t, ALL_CATEGORIES, False, False))
t = best_of(lambda t: tiled(D, t, t, ALL_CATEGORIES, False), THRESHOLDS)
ARMS["B2 + tiles, one shared threshold, truncation filter"] = (t, t, tiled(D, t, t, ALL_CATEGORIES, False))
t = best_of(lambda t: tiled(D, WIN_T, t, ALL_CATEGORIES, False), TILE_THRESHOLDS)
ARMS["B3 + tiles, per-view thresholds, truncation filter"] = (WIN_T, t, tiled(D, WIN_T, t, ALL_CATEGORIES, False))
t = best_of(lambda t: tiled(D, WIN_T, t, ALL_CATEGORIES, True), TILE_THRESHOLDS)
ARMS["B4 + size gate (all 120 categories tiled)"] = (WIN_T, t, tiled(D, WIN_T, t, ALL_CATEGORIES, True))
TILE_T = best_of(lambda t: tiled(D, WIN_T, t, TILE_CATEGORIES, True), TILE_THRESHOLDS)
ARMS["B5 selective tiling: + small/medium category subset"] = (WIN_T, TILE_T, tiled(D, WIN_T, TILE_T, TILE_CATEGORIES, True))

EXP02B = pd.DataFrame([summarize(dets, name, **{"global thr": gt, "tile thr": tt}) for name, (gt, tt, dets) in ARMS.items()])
display(EXP02B[["arm", "global thr", "tile thr", "P", "R", "F1", "R_small", "R_medium", "R_large", "det/img"]].round(4))

tiles_in = [d for k in CAL_KEYS for d in D[k] if not is_global(d) and d.score >= TILE_T and d.category_id in TILE_CATEGORIES]
cut = sum(pdet.touches_inner_edge(d.bbox, pdet.tile_grid(*SIZES[k])[pdet.view_index(d.source)], *SIZES[k])
          for k in CAL_KEYS for d in D[k]
          if not is_global(d) and d.score >= TILE_T and d.category_id in TILE_CATEGORIES)
print(f"tile detections >= {TILE_T} in the tiled subset: {len(tiles_in)} | removed as inner-edge truncation "
      f"artifacts: {cut} ({cut / len(tiles_in):.1%})")
'''),
    ("md", "23_ablation"),
    ("code", r'''
COUNTS_B = {name: per_image_counts(dets) for name, (_, _, dets) in ARMS.items()}
COUNTS_EXP01 = per_image_counts(at_threshold({k: EXP01[k] for k in CAL_KEYS}, 0.175))
FINAL = "B5 selective tiling: + small/medium category subset"
rows = []
for label, a, b in (("B5 selective tiling - B0 full image", COUNTS_B["B0 full image"], COUNTS_B[FINAL]),
                    ("B5 (EXP-02A+B) - EXP-01 baseline", COUNTS_EXP01, COUNTS_B[FINAL]),
                    ("B2 shared threshold - B0 full image", COUNTS_B["B0 full image"],
                     COUNTS_B["B2 + tiles, one shared threshold, truncation filter"])):
    boot = bootstrap_delta(a, b)
    rows += [{"comparison": label, "metric": m, **boot[m]} for m in METRICS]
BOOT_B = pd.DataFrame(rows)
BOOT_B["significant (95% CI excludes 0)"] = (BOOT_B["ci_low"] > 0) | (BOOT_B["ci_high"] < 0)
display(BOOT_B.round(4))

fig, ax = plt.subplots(figsize=(8, 3.6))
arms = [("EXP-01 baseline", COUNTS_EXP01), ("B0 full image", COUNTS_B["B0 full image"]), ("B5 selective tiling", COUNTS_B[FINAL])]
for i, (name, counts) in enumerate(arms):
    m = metrics_from_sums(counts.sum(axis=0))
    vals = [float(m[f"R_{b}"]) for b in BUCKETS]
    bars = ax.bar(np.arange(3) + (i - 1) * 0.27, vals, width=0.27, label=name)
    ax.bar_label(bars, fmt="%.3f", fontsize=7)
ax.set_xticks(range(3), [f"{b}\n(n={n_bucket[b]})" for b in BUCKETS])
ax.set(ylabel="Box recall (IoU >= 0.5)", ylim=(0, 1), title="CAL box recall by GT size bucket")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
s3 = BOOT_B[BOOT_B["comparison"] == "B5 (EXP-02A+B) - EXP-01 baseline"].set_index("metric")
print("S3 check vs EXP-01 on CAL:",
      f"delta R_small = {s3.loc['R_small', 'delta']:+.4f} [{s3.loc['R_small', 'ci_low']:+.4f}, {s3.loc['R_small', 'ci_high']:+.4f}],",
      f"delta Box F1 = {s3.loc['F1', 'delta']:+.4f} [{s3.loc['F1', 'ci_low']:+.4f}, {s3.loc['F1', 'ci_high']:+.4f}]")
'''),
    ("md", "24_significance"),
    ("code", r'''
rows = []
for bb in BACKBONES:
    for v in pdet.VARIANTS:
        Dv, gt_thr = DETS[bb, v], BEST_T[bb, v]
        base_row = summarize(at_threshold(global_only(Dv), gt_thr), "")
        tt = best_of(lambda t: tiled(Dv, gt_thr, t, TILE_CATEGORIES, True), TILE_THRESHOLDS)
        tile_row = summarize(tiled(Dv, gt_thr, tt, TILE_CATEGORIES, True), "")
        rows.append({"arm": f"{bb} / {v}", "global thr": gt_thr, "tile thr": tt, "F1 full": base_row["F1"],
                     "F1 +tiles": tile_row["F1"], "delta F1": tile_row["F1"] - base_row["F1"],
                     "R_small full": base_row["R_small"], "R_small +tiles": tile_row["R_small"],
                     "delta R_small": tile_row["R_small"] - base_row["R_small"],
                     "delta R_large": tile_row["R_large"] - base_row["R_large"]})
ROBUST = pd.DataFrame(rows)
display(ROBUST.round(4))
'''),
    ("md", "25_robustness"),
    ("code", r'''
rows = []
for bb in BACKBONES:
    sec = np.array([r["duration_ms"] for r in TRACE[bb]]) / 1000
    rows.append({"backbone": bb, "GPU": META[bb, "single_raw"]["gpu"], "images": len(sec),
                 "mean s/img (5 views)": sec.mean(), "p95 s/img": np.percentile(sec, 95),
                 "approx. s/view": sec.mean() / 5, "peak VRAM MB": max(r["vram_peak_mb"] for r in TRACE[bb]),
                 "raw global det/img (chunked_alias)": np.mean([sum(is_global(d) for d in DETS[bb, "chunked_alias"][k]) for k in DETS[bb, "chunked_alias"]]),
                 "raw tile det/img (chunked_alias)": np.mean([sum(not is_global(d) for d in DETS[bb, "chunked_alias"][k]) for k in DETS[bb, "chunked_alias"]])})
display(pd.DataFrame(rows).round(2))
'''),
    ("md", "26_cost"),
    ("code", r'''
b0 = ev.box_match_details({k: as_preds(ARMS["B0 full image"][2][k]) for k in CAL_KEYS}, GT)
b5 = ev.box_match_details({k: as_preds(ARMS[FINAL][2][k]) for k in CAL_KEYS}, GT)
rescued = {k: [i for i, (a, b) in enumerate(zip(b0[k]["gt_matched"], b5[k]["gt_matched"])) if b and not a] for k in CAL_KEYS}
lost = {k: [i for i, (a, b) in enumerate(zip(b0[k]["gt_matched"], b5[k]["gt_matched"])) if a and not b] for k in CAL_KEYS}
by_size = Counter(GT_BUCKET[k][i] for k in CAL_KEYS for i in rescued[k])
by_size_lost = Counter(GT_BUCKET[k][i] for k in CAL_KEYS for i in lost[k])
print(f"GT instances rescued by tiling: {sum(map(len, rescued.values()))} {dict(by_size)} | "
      f"lost: {sum(map(len, lost.values()))} {dict(by_size_lost)}")
top = Counter(cfg.CATEGORIES[GT[k]['categories'][i]] for k in CAL_KEYS for i in rescued[k]).most_common(12)
print("most rescued categories:", ", ".join(f"{n} ({c})" for n, c in top))

QUAL = max(CAL_KEYS, key=lambda k: (len(rescued[k]), k))
W, H = SIZES[QUAL]
print(f"{QUAL}: rescued " + ", ".join(
    f"{cfg.CATEGORIES[GT[QUAL]['categories'][i]]} (area_ratio {AREA_RATIO[QUAL][i]:.5f})" for i in rescued[QUAL]))
fig, ax = plt.subplots(figsize=(8, 8 * H / W))
ax.imshow(Image.open(PATHS.dev_images_dir / f"{QUAL}.jpg"))
for (x1, y1, x2, y2) in pdet.tile_grid(W, H):
    ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1, ls=":", ec="white"))
for i, (box, cid, hit) in enumerate(zip(GT[QUAL]["boxes"], GT[QUAL]["categories"], b5[QUAL]["gt_matched"])):
    color = "lime" if i in rescued[QUAL] else ("deepskyblue" if hit else "red")
    ax.add_patch(mpatches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, lw=1.6, ec=color))
    if i in rescued[QUAL]:
        ax.text(box[0], box[1] - 2, cfg.CATEGORIES[cid], color="lime", fontsize=7, weight="bold")
ax.set(title=f"{QUAL}: GT boxes - lime = rescued by tiling, blue = found by both, red = still missed", xticks=[], yticks=[])
ax.grid(False)
plt.show()
'''),
    ("md", "27_qualitative"),
    ("code", r'''
final = EXP02B.set_index("arm").loc[FINAL]
SUMMARY = {
    "exp02a_selection": {"rule": "max CAL Box F1 over 8 global arms", "backbone": WIN_KEY[0], "variant": WIN_KEY[1],
                         "global_threshold": WIN_T},
    "exp02b_selective_tiling": {"global_threshold": WIN_T, "tile_threshold": TILE_T,
                                "tile_max_area_ratio": cfg.MEDIUM_MAX_AREA_RATIO,
                                "tile_categories": str(TILING_PATH.relative_to(REPO)),
                                "truncation_filter_margin_px": pdet.TILE_EDGE_MARGIN, "overlap": pdet.TILE_OVERLAP},
    "cal_metrics": {k: round(float(final[k]), 4) for k in METRICS},
    "s3_vs_exp01": {m: {k: round(float(s3.loc[m, k]), 4) for k in ("delta", "ci_low", "ci_high")} for m in ("F1", "R_small")},
    "thresholds_selected_on": "CAL (in-sample); VAL held out for T4 Section 3",
}
SUMMARY_PATH = CACHE_DIR / "exp02ab_summary.json"
SUMMARY_PATH.write_text(json.dumps(SUMMARY, indent=2), encoding="utf-8")
print(json.dumps(SUMMARY, indent=2))
'''),
    ("md", "30_conclusions"),
]


def _cell(kind: str, src: str):
    if kind == "md":
        cell = new_markdown_cell(md(src))
        cell.metadata["analysis"] = src  # lets --refresh-markdown find it again in an executed copy
        return cell
    return new_code_cell(src.strip())


def build() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (TPI .venv)", "language": "python"}
    nb.cells = [_cell(kind, src) for kind, src in CELLS + EXP02C_CELLS]
    return nb


def refresh_markdown(path: Path) -> nbformat.NotebookNode:
    """Replace the analysis Markdown of an already executed notebook, keeping every code cell and output as is."""
    nb = nbformat.read(path, as_version=4)
    expected = [src for kind, src in CELLS + EXP02C_CELLS if kind == "md"]
    found = [c.metadata.get("analysis") for c in nb.cells if c.cell_type == "markdown"]
    if found != expected:
        raise ValueError(f"{path} does not have this builder's cell layout; rebuild and execute it instead")
    for c in nb.cells:
        if c.cell_type == "markdown":
            c.source = md(c.metadata["analysis"])
    return nb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--refresh-markdown", action="store_true")
    parser.add_argument("--output", type=Path, default=NOTEBOOK)
    args = parser.parse_args(argv)
    if args.refresh_markdown:
        nb = refresh_markdown(args.output)
    else:
        nb = build()
        if args.execute:
            from nbclient import NotebookClient

            NotebookClient(nb, timeout=3600, kernel_name="python3",
                           resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
