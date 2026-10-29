"""Cells of notebooks/02_Perception_Ensemble_Tiling_EXP02.ipynb, Sections 3-5 (EXP-02C, T4 / S4).

Appended by scripts/build_notebook_02.py after Sections 1-2 and executed in the same kernel, so they reuse its
helpers (sweep, summarize, per_image_counts, bootstrap_delta, tiled, ...) and selections (WIN_KEY, WIN_T, TILE_T,
TILE_CATEGORIES). Every hyperparameter is chosen on CAL; VAL is only reported.

With TPI_EXP02C_DRY_RUN=1 the cells run on CPU with OWLv2 Large standing in for MM-GDINO and bbox-fill masks
instead of SAM2, writing to artifacts/cache_exp02/exp02c_dryrun/: a code-path check, never a result.
"""

EXP02C_CELLS: list[tuple[str, str]] = [
    ("md", "40_section3"),
    ("code", r'''
import csv, os, subprocess, time
from src import detection_fusion as fu
from src import qa_engine as qa
from src import scene_builder as sb
from scripts.run_exp02c_segmentation import config_sha

DRY_RUN = os.environ.get("TPI_EXP02C_DRY_RUN") == "1"
EXP02C_DIR = CACHE_DIR / ("exp02c_dryrun" if DRY_RUN else "exp02c")
GDINO_DIR = CACHE_DIR / ("large" if DRY_RUN else "mmgdino_large")
if DRY_RUN:
    print("DRY RUN: 'mmgdino' is OWLv2 Large retagged and masks are bbox fills -> no number below is a result")
GD_VARIANTS = [v for v in ("chunked_raw", "chunked_alias") if (GDINO_DIR / f"detections_{v}.json.gz").exists()]
if "chunked_raw" not in GD_VARIANTS:
    raise FileNotFoundError(f"{GDINO_DIR.relative_to(REPO)}/detections_chunked_raw.json.gz is missing: run "
                            "scripts/run_exp02_inference.py --backbone mmgdino_large on a GPU "
                            "(notebooks/kaggle/T4_EXP02C_runner.ipynb)")
for v in GD_VARIANTS:
    raw_gd, META["mmgdino", v] = pl.load_detection_cache(GDINO_DIR / f"detections_{v}.json.gz")
    DETS["mmgdino", v] = {k: fu.retag(x, "mmgdino") for k, x in raw_gd.items()}
    missing = sorted(set(SIZES) - set(DETS["mmgdino", v]))
    assert not missing, f"MM-GDINO {v} cache is incomplete: {len(missing)} images missing (re-run the resumable runner)"
display(pd.DataFrame([{"variant": v, "model": m["model"], "device": m["device"], "gpu": m["gpu"],
                       "created_at": m["created_at"], "images": len(DETS["mmgdino", v]), "score floor": m["min_score"],
                       "max text tokens per block": max(m.get("n_text_tokens", {}).values(), default=None),
                       "raw global det/img": np.mean([sum(is_global(d) for d in DETS["mmgdino", v][k]) for k in SIZES]),
                       "raw tile det/img": np.mean([sum(not is_global(d) for d in DETS["mmgdino", v][k]) for k in SIZES])}
                      for v in GD_VARIANTS for m in [META["mmgdino", v]]]).round(2))
'''),
    ("md", "41_setup"),
    ("code", r'''
GD_THRESHOLDS = np.round(np.arange(0.10, 0.801, 0.025), 3)
rows, GD_BEST = [], {}
for v in GD_VARIANTS:
    Dg = DETS["mmgdino", v]
    g = global_only(Dg)
    sw = sweep(lambda t: at_threshold(g, t), GD_THRESHOLDS)
    gt = float(sw.loc[sw["F1"].idxmax(), "threshold"])
    tt = best_of(lambda t: tiled(Dg, gt, t, TILE_CATEGORIES, True), GD_THRESHOLDS)
    GD_BEST[v] = (gt, tt)
    rows.append(summarize(at_threshold(g, gt), f"MM-GDINO {v}: full image", **{"global thr": gt, "tile thr": None}))
    rows.append(summarize(tiled(Dg, gt, tt, TILE_CATEGORIES, True), f"MM-GDINO {v}: + selective tiling",
                          **{"global thr": gt, "tile thr": tt}))
rows.append(summarize(ARMS["B0 full image"][2], "OWLv2 Base chunked_raw: full image (EXP-02A)",
                      **{"global thr": WIN_T, "tile thr": None}))
rows.append(summarize(ARMS[FINAL][2], "OWLv2 Base chunked_raw: + selective tiling (EXP-02B)",
                      **{"global thr": WIN_T, "tile thr": TILE_T}))
GD_TABLE = pd.DataFrame(rows)
display(GD_TABLE[["arm", "global thr", "tile thr", "P", "R", "F1", "R_small", "R_medium", "R_large", "det/img"]].round(4))
GD_VARIANT = max(GD_VARIANTS, key=lambda v: GD_TABLE.set_index("arm").loc[f"MM-GDINO {v}: + selective tiling", "F1"])
GD_T, GD_TILE_T = GD_BEST[GD_VARIANT]

# Complementarity at each family's own operating point (CAL): which GT instances each family finds.
owl_hit = ev.box_match_details({k: as_preds(ARMS[FINAL][2][k]) for k in CAL_KEYS}, GT)
gd_op = tiled(DETS["mmgdino", GD_VARIANT], GD_T, GD_TILE_T, TILE_CATEGORIES, True)
gd_hit = ev.box_match_details({k: as_preds(gd_op[k]) for k in CAL_KEYS}, GT)
comp = Counter()
for k in CAL_KEYS:
    for a, b, size in zip(owl_hit[k]["gt_matched"], gd_hit[k]["gt_matched"], GT_BUCKET[k]):
        comp[size, "both" if a and b else "OWLv2 only" if a else "MM-GDINO only" if b else "neither"] += 1
COMP = pd.DataFrame([{"size": s, **{c: comp[s, c] for c in ("both", "OWLv2 only", "MM-GDINO only", "neither")}}
                     for s in BUCKETS])
COMP["oracle union recall"] = (COMP["both"] + COMP["OWLv2 only"] + COMP["MM-GDINO only"]) / COMP.drop(columns="size").sum(axis=1)
display(COMP.round(3))
boot = bootstrap_delta(per_image_counts(ARMS[FINAL][2]), per_image_counts(gd_op))
print(f"selected MM-GDINO variant: {GD_VARIANT} (global {GD_T}, tiles {GD_TILE_T}) | MM-GDINO - OWLv2 Base (both tiled), "
      f"CAL: " + ", ".join(f"d{m} {boot[m]['delta']:+.4f} [{boot[m]['ci_low']:+.4f}, {boot[m]['ci_high']:+.4f}]"
                           for m in ("F1", "P", "R", "R_small")))
'''),
    ("md", "42_gdino_alone"),
    ("code", r'''
# Per-family, per-view calibration: each view's CAL F1-optimal threshold tau maps to the canonical score 0.5.
assert WIN_KEY == ("base", "chunked_raw"), WIN_KEY
LARGE_T = BEST_T["large", "chunked_raw"]
LARGE_TILE_T = best_of(lambda t: tiled(DETS["large", "chunked_raw"], LARGE_T, t, TILE_CATEGORIES, True), TILE_THRESHOLDS)
rel = lambda p: str(p.relative_to(REPO))
MEMBERS = {
    "owlv2b": {"spec": fu.MemberSpec("owlv2", pdet.OWLV2_BASE, "chunked_raw",
                                     rel(CACHE_DIR / "base" / "detections_chunked_raw.json.gz")),
               "dets": {k: fu.retag(x, "owlv2b") for k, x in DETS["base", "chunked_raw"].items()},
               "tau": {"global": WIN_T, "tile": TILE_T}, "floor": META["base", "chunked_raw"]["min_score"]},
    "owlv2l": {"spec": fu.MemberSpec("owlv2", pdet.OWLV2_LARGE, "chunked_raw",
                                     rel(CACHE_DIR / "large" / "detections_chunked_raw.json.gz")),
               "dets": {k: fu.retag(x, "owlv2l") for k, x in DETS["large", "chunked_raw"].items()},
               "tau": {"global": LARGE_T, "tile": LARGE_TILE_T}, "floor": META["large", "chunked_raw"]["min_score"]},
    "mmgdino": {"spec": fu.MemberSpec("owlv2" if DRY_RUN else "gdino", pdet.OWLV2_LARGE if DRY_RUN else pdet.MM_GDINO_LARGE,
                                      GD_VARIANT, rel(GDINO_DIR / f"detections_{GD_VARIANT}.json.gz")),
                "dets": DETS["mmgdino", GD_VARIANT], "tau": {"global": GD_T, "tile": GD_TILE_T},
                "floor": META["mmgdino", GD_VARIANT]["min_score"]},
}
TILE_SET = set(TILE_CATEGORIES)


def view_scores(name, view):
    """CAL raw scores of one member view, tiles after the same filter the fusion applies."""
    out = []
    for k in CAL_KEYS:
        dets = MEMBERS[name]["dets"][k]
        if view == "tile":
            dets = pdet.filter_tile_views(dets, *SIZES[k], TILE_SET, tile_max_area_ratio=cfg.MEDIUM_MAX_AREA_RATIO)
        out += [d.score for d in dets if is_global(d) == (view == "global")]
    return np.array(out)


CALIBRATORS, rows = {}, []
for name, m in MEMBERS.items():
    for view in ("global", "tile"):
        s = view_scores(name, view)
        CALIBRATORS[f"{name}:{view}"] = fu.QuantileCalibrator.fit(s, m["tau"][view], m["floor"])
        rows.append({"member view": f"{name}:{view}", "CAL scores": len(s), "floor": m["floor"], "tau (CAL F1-opt)": m["tau"][view],
                     "share >= tau": float(np.mean(s >= m["tau"][view])), "p50 raw": float(np.median(s)),
                     "p99 raw": float(np.percentile(s, 99)), "knots": len(CALIBRATORS[f"{name}:{view}"].knots_x)})
display(pd.DataFrame(rows).round(4))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for name in MEMBERS:
    axes[0].hist(view_scores(name, "global"), bins=60, histtype="step", lw=1.4, label=f"{name}:global", log=True)
    for view, ls in (("global", "-"), ("tile", "--")):
        c = CALIBRATORS[f"{name}:{view}"]
        x = np.linspace(c.knots_x[0], c.knots_x[-1], 300)
        axes[1].plot(x, c(x), ls=ls, label=f"{name}:{view}")
        axes[1].plot([MEMBERS[name]["tau"][view]], [0.5], "k.", ms=6)
axes[0].set(title="CAL raw score distribution (global view, log count)", xlabel="raw detector score", ylabel="detections")
axes[1].axhline(0.5, color="gray", lw=0.8)
axes[1].set(title="Monotone quantile calibration: tau -> 0.5 (dots)", xlabel="raw score", ylabel="canonical score",
            xlim=(0, 0.8))
for ax in axes:
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
'''),
    ("md", "43_calibration"),
    ("code", r'''
IDENTITY = fu.QuantileCalibrator((0.0, 1.0), (0.0, 1.0))
FUSION_GRID = np.round(np.arange(0.10, 0.951, 0.025), 3)


def make_config(names, **kw):
    params = dict(members={n: MEMBERS[n]["spec"] for n in names},
                  calibrators={k: c for k, c in CALIBRATORS.items() if fu.detector_of(k) in names},
                  tile_category_ids=frozenset(TILE_CATEGORIES), tile_max_area_ratio=cfg.MEDIUM_MAX_AREA_RATIO,
                  group_of={}, thresholds={g: fu.CANONICAL_SCORE for g in fu.GROUPS})
    params.update(kw)
    return fu.EnsembleConfig(**params)


def fused(names, keys=None, **kw):
    """Unthresholded fused detections per image (thresholds are swept afterwards on CAL)."""
    c = make_config(names, **kw)
    return {k: fu.fuse_views({n: MEMBERS[n]["dets"][k] for n in names}, *SIZES[k], c) for k in (keys or DEV_KEYS)}


def cut(fused_by_key, thr, keys=None):
    return {k: [d for d in fused_by_key[k] if d.score >= thr][:pdet.MAX_DETECTIONS] for k in (keys or CAL_KEYS)}


def grouped(fused_by_key, group_of, thresholds, keys=None):
    return {k: fu.apply_group_thresholds(fused_by_key[k], group_of, thresholds)[:pdet.MAX_DETECTIONS]
            for k in (keys or CAL_KEYS)}


def tune(fused_by_key):
    sw = sweep(lambda t: cut(fused_by_key, t), FUSION_GRID)
    return float(sw.loc[sw["F1"].idxmax(), "threshold"])


# Sanity: one member, calibrated, NMS at the canonical 0.5 vs its raw per-view thresholds (EXP-02B rule). The keep
# rule is identical (monotone map, tau -> 0.5); only the cross-view NMS ranking changes (calibrated vs raw score).
rows = []
for name, raw_arm in (("owlv2b", ARMS[FINAL][2]), ("mmgdino", gd_op)):
    one = cut(fused([name], fusion="nms"), fu.CANONICAL_SCORE)
    rows.append({"member": name, "F1 raw per-view thresholds + NMS": summarize(raw_arm, "")["F1"],
                 "F1 calibrated, NMS, cut at 0.5": summarize(one, "")["F1"],
                 "det/img raw": np.mean([len(raw_arm[k]) for k in CAL_KEYS]),
                 "det/img calibrated": np.mean([len(one[k]) for k in CAL_KEYS])})
display(pd.DataFrame(rows).round(4))
'''),
    ("md", "44_calibration_check"),
    ("md", "50_section4"),
    ("code", r'''
PAIR = ["owlv2b", "mmgdino"]
SPECS = {
    "C1 OWLv2 Base alone (calibrated, NMS)": (["owlv2b"], {"fusion": "nms"}),
    "C2 MM-GDINO alone (calibrated, NMS)": (["mmgdino"], {"fusion": "nms"}),
    "C3 raw-score union + NMS (no calibration)": (PAIR, {"fusion": "nms", "calibrators": {
        f"{n}:{v}": IDENTITY for n in PAIR for v in ("global", "tile")}}),
    "C4 calibrated union + NMS": (PAIR, {"fusion": "nms"}),
    "C5 calibrated WBF, no self-collision rule": (PAIR, {"anti_collision": False}),
    "C6 calibrated WBF + self-collision rule": (PAIR, {}),
    "C6b OWLv2 Large + MM-GDINO (WBF + rule)": (["owlv2l", "mmgdino"], {}),
    "C6c OWLv2 Base + Large + MM-GDINO (WBF + rule)": (["owlv2b", "owlv2l", "mmgdino"], {}),
}
t0 = time.perf_counter()
FUSED = {name: fused(members, **kw) for name, (members, kw) in SPECS.items()}
FUSE_SEC = (time.perf_counter() - t0) / (len(SPECS) * len(DEV_KEYS))
THR_C, rows = {}, []
for name, (members, kw) in SPECS.items():
    THR_C[name] = tune(FUSED[name])
    rows.append(summarize(cut(FUSED[name], THR_C[name]), name, members="+".join(members), threshold=THR_C[name]))
ABL = pd.DataFrame(rows)
display(ABL[["arm", "members", "threshold", "P", "R", "F1", "R_small", "R_medium", "R_large", "det/img"]].round(4))
WBF_ARMS = [n for n in SPECS if n.startswith("C6")]
SEL = max(WBF_ARMS, key=lambda n: (ABL.set_index("arm").loc[n, "F1"], -len(SPECS[n][0])))
SEL_MEMBERS = SPECS[SEL][0]
print(f"member-set selection (max CAL F1 over the WBF + rule arms, fewer members on ties): {SEL} -> {SEL_MEMBERS} | "
      f"offline fusion cost {1000 * FUSE_SEC:.1f} ms/img (CPU)")
COUNTS_C = {name: per_image_counts(cut(FUSED[name], THR_C[name])) for name in SPECS}
rows = []
for label, a, b in (("C4 calibrated - C3 raw union (NMS)", "C3 raw-score union + NMS (no calibration)", "C4 calibrated union + NMS"),
                    ("C6 WBF + rule - C4 NMS", "C4 calibrated union + NMS", "C6 calibrated WBF + self-collision rule"),
                    ("C6 rule - C5 no rule", "C5 calibrated WBF, no self-collision rule", "C6 calibrated WBF + self-collision rule"),
                    ("C6 ensemble - C1 OWLv2 alone", "C1 OWLv2 Base alone (calibrated, NMS)", "C6 calibrated WBF + self-collision rule")):
    boot = bootstrap_delta(COUNTS_C[a], COUNTS_C[b])
    rows += [{"comparison": label, "metric": m, **boot[m]} for m in ("F1", "P", "R", "R_small")]
BOOT_C = pd.DataFrame(rows)
BOOT_C["significant (95% CI excludes 0)"] = (BOOT_C["ci_low"] > 0) | (BOOT_C["ci_high"] < 0)
display(BOOT_C.round(4))
'''),
    ("md", "51_ablation"),
    ("code", r'''
# What the non-self-collision rule changes (C5 -> C6, same members and calibration, each at its CAL threshold).
c5_name, c6_name = "C5 calibrated WBF, no self-collision rule", "C6 calibrated WBF + self-collision rule"
c5 = ev.box_match_details({k: as_preds(cut(FUSED[c5_name], THR_C[c5_name])[k]) for k in CAL_KEYS}, GT)
c6 = ev.box_match_details({k: as_preds(cut(FUSED[c6_name], THR_C[c6_name])[k]) for k in CAL_KEYS}, GT)
kept_apart = {k: [i for i, (a, b) in enumerate(zip(c5[k]["gt_matched"], c6[k]["gt_matched"])) if b and not a] for k in CAL_KEYS}
merged_away = {k: [i for i, (a, b) in enumerate(zip(c5[k]["gt_matched"], c6[k]["gt_matched"])) if a and not b] for k in CAL_KEYS}
print(f"GT instances matched only with the rule: {sum(map(len, kept_apart.values()))} | only without it: "
      f"{sum(map(len, merged_away.values()))} | TP/FP with rule {sum(v['tp'] for v in c6.values())}/"
      f"{sum(v['fp'] for v in c6.values())} vs without {sum(v['tp'] for v in c5.values())}/{sum(v['fp'] for v in c5.values())}")
top = Counter(cfg.CATEGORIES[GT[k]["categories"][i]] for k in CAL_KEYS for i in kept_apart[k]).most_common(10)
print("categories whose adjacent instances the rule keeps apart:", ", ".join(f"{n} ({c})" for n, c in top) or "none")
'''),
    ("md", "52_anticollision"),
    ("code", r'''
# LVIS sensitivity groups (CAL only, frozen): micro = CAL GT median area_ratio < 0.0035; fp_prone = >= 3 verified
# false positives (category in the image's neg_category_ids) and >= 50% of its evaluated predictions at the selected
# operating point; everything else (and categories absent from CAL) = standard.
MICRO = {c for c in cfg.CATEGORIES if c in by_cat and np.median(by_cat[c]) < cfg.SMALL_MAX_AREA_RATIO}
op = cut(FUSED[SEL], THR_C[SEL])
neg_fp, evaluated = Counter(), Counter()
for k in CAL_KEYS:
    for d in op[k]:
        neg_fp[d.category_id] += d.category_id in GT[k]["neg"]
        evaluated[d.category_id] += d.category_id in GT[k]["neg"] or d.category_id in GT[k]["pos"]
FP_PRONE = fu.select_fp_prone(neg_fp, evaluated, min_count=3, min_share=0.5)
GROUP_OF = fu.assign_sensitivity_groups(sorted(cfg.CATEGORIES), MICRO, FP_PRONE)

THR = {g: THR_C[SEL] for g in fu.GROUPS}
history = []
for rnd in (1, 2):  # coordinate ascent on CAL, ties broken towards the current value
    for g in fu.GROUPS:
        if g not in GROUP_OF.values():
            continue
        f1, _, t = max((summarize(grouped(FUSED[SEL], GROUP_OF, {**THR, g: float(t)}), "")["F1"], -abs(t - THR[g]), float(t))
                       for t in FUSION_GRID)
        THR[g] = t
        history.append({"round": rnd, "group": g, "threshold": t, "CAL F1": f1})
display(pd.DataFrame(history).round(4))
display(pd.DataFrame([{"group": g, "threshold": THR[g], "categories": sum(v == g for v in GROUP_OF.values()),
                       "CAL GT instances": sum(GROUP_OF[c] == g for k in CAL_KEYS for c in GT[k]["categories"]),
                       "members": ", ".join(cfg.CATEGORIES[c] for c in sorted(GROUP_OF) if GROUP_OF[c] == g)[:160]}
                      for g in fu.GROUPS]))

C7 = f"C7 = {SEL.split(' ')[0]} + sensitivity-group thresholds"
FUSED[C7], THR_C[C7] = FUSED[SEL], None
COUNTS_C[C7] = per_image_counts(grouped(FUSED[SEL], GROUP_OF, THR))
boot = bootstrap_delta(COUNTS_C[SEL], COUNTS_C[C7])
print(f"{C7} - {SEL.split(' ')[0]} on CAL: " + ", ".join(
    f"d{m} {boot[m]['delta']:+.4f} [{boot[m]['ci_low']:+.4f}, {boot[m]['ci_high']:+.4f}]" for m in ("F1", "P", "R", "R_small")))

EXP02C_CONFIG = make_config(SEL_MEMBERS, group_of=GROUP_OF, thresholds=THR, notes={
    "selected_on": "CAL (80 images); VAL never used for selection", "member_arm": SEL,
    "member_taus": {n: MEMBERS[n]["tau"] for n in SEL_MEMBERS}, "dry_run": DRY_RUN,
    "fp_prone": sorted(cfg.CATEGORIES[c] for c in FP_PRONE), "micro_categories": len(MICRO)})
CONFIG_PATH = EXP02C_CONFIG.save(EXP02C_DIR / "config.json")
# The frozen JSON config through the blind path must reproduce the tuned CAL detections exactly.
reloaded = fu.EnsembleConfig.load(CONFIG_PATH)
replay = {k: fu.fuse_image({n: MEMBERS[n]["dets"][k] for n in SEL_MEMBERS}, *SIZES[k], reloaded) for k in CAL_KEYS}
assert replay == grouped(FUSED[SEL], GROUP_OF, THR), "frozen config does not reproduce the tuned detections"
print(f"wrote {CONFIG_PATH.relative_to(REPO)} (sha {config_sha(CONFIG_PATH)}); blind replay on CAL: identical")
'''),
    ("md", "53_groups"),
    ("code", r'''
EXP01_NAME, A_NAME, B_NAME, C_NAME = ("EXP-01 baseline", "EXP-02A chunked prompting", "EXP-02B + selective tiling",
                                      "EXP-02C calibrated ensemble + anti-collision WBF")
FINAL_ARMS = {
    EXP01_NAME: at_threshold(EXP01, 0.175, DEV_KEYS),
    A_NAME: at_threshold(global_only(DETS[WIN_KEY], DEV_KEYS), WIN_T, DEV_KEYS),
    B_NAME: tiled(DETS[WIN_KEY], WIN_T, TILE_T, TILE_CATEGORIES, True, keys=DEV_KEYS),
    C_NAME: grouped(FUSED[SEL], GROUP_OF, THR, DEV_KEYS),
}


def metric_ci(counts, n_boot=1000, seed=cfg.SEED):
    """Point estimate and 95% percentile CI of each metric (image-level bootstrap)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    m, point = metrics_from_sums(counts[idx].sum(axis=1)), metrics_from_sums(counts.sum(axis=0))
    return {k: (float(point[k]), float(np.percentile(m[k], 2.5)), float(np.percentile(m[k], 97.5))) for k in METRICS}


COUNTS_F, rows = {}, []
for split, keys in (("CAL", CAL_KEYS), ("VAL", VAL_KEYS)):
    for name, dets in FINAL_ARMS.items():
        COUNTS_F[split, name] = per_image_counts(dets, keys)
        ci = metric_ci(COUNTS_F[split, name])
        rows.append({"split": split, "arm": name, **{m: f"{v[0]:.4f} [{v[1]:.3f}, {v[2]:.3f}]" for m, v in ci.items()
                                                     if m in ("P", "R", "F1", "R_small")},
                     "det/img": round(float(np.mean([len(dets[k]) for k in keys])), 1)})
COMPARISON = pd.DataFrame(rows)
display(COMPARISON)

rows = []
for split in ("CAL", "VAL"):
    for ref in (EXP01_NAME, B_NAME):
        boot = bootstrap_delta(COUNTS_F[split, ref], COUNTS_F[split, C_NAME])
        rows += [{"split": split, "comparison": f"EXP-02C - {ref.split(' ')[0]}", "metric": m, **boot[m]}
                 for m in ("F1", "P", "R", "R_small")]
DELTAS = pd.DataFrame(rows)
DELTAS["significant (95% CI excludes 0)"] = (DELTAS["ci_low"] > 0) | (DELTAS["ci_high"] < 0)
display(DELTAS.round(4))
F1_BY = {(s, n): float(metrics_from_sums(COUNTS_F[s, n].sum(axis=0))["F1"]) for s in ("CAL", "VAL") for n in FINAL_ARMS}
display(pd.DataFrame([{"arm": n, "CAL F1": F1_BY["CAL", n], "VAL F1": F1_BY["VAL", n],
                       "CAL - VAL (optimism)": F1_BY["CAL", n] - F1_BY["VAL", n]} for n in FINAL_ARMS]).round(4))
s4 = DELTAS[(DELTAS.split == "VAL") & (DELTAS.metric == "F1")].set_index("comparison")
print(f"S4 check (VAL, held out): Box F1 EXP-02C = {F1_BY['VAL', C_NAME]:.4f} (bar 0.60: "
      f"{'met' if F1_BY['VAL', C_NAME] > 0.60 else 'NOT met'}) | " + " | ".join(
          f"{c}: {r['delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]" for c, r in s4.iterrows()))
'''),
    ("md", "54_comparison"),
    ("code", r'''
mb = {n: ev.box_match_details({k: as_preds(FINAL_ARMS[n][k]) for k in DEV_KEYS}, GT) for n in (B_NAME, C_NAME)}
rescued_small = {k: [i for i, (a, b) in enumerate(zip(mb[B_NAME][k]["gt_matched"], mb[C_NAME][k]["gt_matched"]))
                     if b and not a and GT_BUCKET[k][i] == "small"] for k in DEV_KEYS}
neg_of = lambda name, k: [d for d in FINAL_ARMS[name][k] if d.category_id in GT[k]["neg"]]
c6_all = cut(FUSED[c6_name], THR_C[c6_name], DEV_KEYS)
c5_all = cut(FUSED[c5_name], THR_C[c5_name], DEV_KEYS)
m5 = ev.box_match_details({k: as_preds(c5_all[k]) for k in DEV_KEYS}, GT)
m6 = ev.box_match_details({k: as_preds(c6_all[k]) for k in DEV_KEYS}, GT)
apart = {k: [i for i, (a, b) in enumerate(zip(m5[k]["gt_matched"], m6[k]["gt_matched"])) if b and not a] for k in DEV_KEYS}
K_MICRO = max(VAL_KEYS, key=lambda k: (len(rescued_small[k]), k))
K_NEG = max(VAL_KEYS, key=lambda k: (len(neg_of(B_NAME, k)) - len(neg_of(C_NAME, k)), k))
K_ADJ = max(DEV_KEYS, key=lambda k: (len(apart[k]), k))
print(f"VAL: small GT matched by EXP-02C but not EXP-02B: {sum(len(rescued_small[k]) for k in VAL_KEYS)} | "
      f"verified FPs (neg categories) EXP-02B {sum(len(neg_of(B_NAME, k)) for k in VAL_KEYS)} -> EXP-02C "
      f"{sum(len(neg_of(C_NAME, k)) for k in VAL_KEYS)} | DEV: GT kept apart by the rule: {sum(map(len, apart.values()))}")


def draw(ax, key, title):
    ax.imshow(Image.open(PATHS.dev_images_dir / f"{key}.jpg"))
    ax.set(title=title, xticks=[], yticks=[])
    ax.grid(False)


def rect(ax, box, color, lw=1.5, ls="-", label=None):
    ax.add_patch(mpatches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, lw=lw, ls=ls, ec=color))
    if label:
        ax.text(box[0], box[1] - 2, label, color=color, fontsize=6, weight="bold")


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
draw(axes[0], K_MICRO, f"{K_MICRO} (VAL): micro-objects rescued by EXP-02C (lime) vs EXP-02B")
for i in rescued_small[K_MICRO]:
    rect(axes[0], GT[K_MICRO]["boxes"][i], "lime", label=cfg.CATEGORIES[GT[K_MICRO]["categories"][i]])
draw(axes[1], K_NEG, f"{K_NEG} (VAL): EXP-02B false positives on neg_category_ids (red) / kept by EXP-02C (yellow)")
for d in neg_of(B_NAME, K_NEG):
    rect(axes[1], d.bbox, "red", ls="--", label=cfg.CATEGORIES[d.category_id])
for d in neg_of(C_NAME, K_NEG):
    rect(axes[1], d.bbox, "yellow", lw=2)
draw(axes[2], K_ADJ, f"{K_ADJ}: adjacent instances kept apart by the rule (lime) / WBF boxes (blue)")
for d in c6_all[K_ADJ]:
    if d.category_id in {GT[K_ADJ]["categories"][i] for i in apart[K_ADJ]}:
        rect(axes[2], d.bbox, "deepskyblue", lw=1)
for i in apart[K_ADJ]:
    rect(axes[2], GT[K_ADJ]["boxes"][i], "lime", label=cfg.CATEGORIES[GT[K_ADJ]["categories"][i]])
for ax in axes:
    ax.title.set_fontsize(8)
plt.tight_layout()
plt.show()
'''),
    ("md", "55_visuals"),
    ("md", "60_section5"),
    ("code", r'''
SHA = config_sha(CONFIG_PATH)
SCENES_PATH, SEG_META_PATH = EXP02C_DIR / "scenes.jsonl", EXP02C_DIR / "scenes.meta.json"


def scenes_ready():
    if not (SCENES_PATH.exists() and SEG_META_PATH.exists()):
        return False
    ids = {json.loads(l)["image"]["id"] for l in SCENES_PATH.read_text(encoding="utf-8").splitlines() if l}
    return json.loads(SEG_META_PATH.read_text())["config_sha"] == SHA and set(SIZES) <= ids


if not scenes_ready():
    import torch
    if not (DRY_RUN or torch.cuda.is_available()):
        raise RuntimeError(f"{SCENES_PATH.relative_to(REPO)} is missing or stale for config {SHA}: run this notebook on "
                           "a GPU (notebooks/kaggle/T4_EXP02C_runner.ipynb) or scripts/run_exp02c_segmentation.py")
    cmd = [sys.executable, str(REPO / "scripts" / "run_exp02c_segmentation.py"), "--config", str(CONFIG_PATH)]
    res = subprocess.run(cmd + (["--box-masks"] if DRY_RUN else []), cwd=REPO, capture_output=True, text=True)
    print("\n".join(res.stdout.splitlines()[-3:]))
    if res.returncode:
        raise RuntimeError(res.stderr[-4000:])
SCENES = {s["image"]["id"]: s for s in map(json.loads, SCENES_PATH.read_text(encoding="utf-8").splitlines())}
SEG_META = json.loads(SEG_META_PATH.read_text())
assert SEG_META["config_sha"] == SHA and len(SCENES) == len(SIZES)
diff = [k for k in DEV_KEYS if [o["category_id"] for o in SCENES[k]["objects"]] != [d.category_id for d in FINAL_ARMS[C_NAME][k]]]
assert not diff, f"scenes disagree with the frozen EXP-02C boxes on {len(diff)} images"
n_fallback = sum("mask_fallback" in o for s in SCENES.values() for o in s["objects"])
print({k: SEG_META.get(k) for k in ("segmenter", "segmenter_model", "multimask_output", "device", "gpu", "created_at")})
print(f"scenes: {len(SCENES)} | objects: {sum(len(s['objects']) for s in SCENES.values())} | bbox mask fallbacks: "
      f"{n_fallback} | boxes identical to the frozen EXP-02C detections on all {len(DEV_KEYS)} dev images")
'''),
    ("md", "61_scenes"),
    ("code", r'''
EXP01_SCENES_PATH = REPO / "artifacts" / "cache_exp01" / "scenes_thr0.175.jsonl"
EXP01_SCENES = ({s["image"]["id"]: s for s in map(json.loads, EXP01_SCENES_PATH.read_text(encoding="utf-8").splitlines())}
                if EXP01_SCENES_PATH.exists() else None)


def scene_counts(scenes, keys):
    """[n_images, 6]: box tp/fp/fn and mask tp/fp/fn (federated, IoU >= 0.5) per image."""
    rows = []
    for k in keys:
        r = ev.evaluate_federated({k: scenes[k]["objects"]}, GT)
        rows.append([r["box"][x] for x in ("tp", "fp", "fn")] + [r["mask"][x] for x in ("tp", "fp", "fn")])
    return np.array(rows, dtype=float)


def f1s(s):
    return 2 * s[..., 0] / np.maximum(2 * s[..., 0] + s[..., 1] + s[..., 2], 1)


def score_parts(s):
    box, mask = f1s(s[..., :3]), f1s(s[..., 3:])
    return {"Box F1": box, "Mask F1": mask, "0.40 Box + 0.35 Mask": 0.40 * box + 0.35 * mask}


SEG_ARMS = {"EXP-02C + SAM2.1 Base+": SCENES}
if EXP01_SCENES is not None:
    SEG_ARMS = {"EXP-01 + SAM2.1 Base+": EXP01_SCENES, **SEG_ARMS}
rows, deltas, SEG_POINT = [], [], {}
rng = np.random.default_rng(cfg.SEED)
for split, keys in (("CAL", CAL_KEYS), ("VAL", VAL_KEYS)):
    idx = rng.integers(0, len(keys), size=(1000, len(keys)))
    counts = {n: scene_counts(sc, keys) for n, sc in SEG_ARMS.items()}
    for n, c in counts.items():
        point, boot = score_parts(c.sum(axis=0)), score_parts(c[idx].sum(axis=1))
        SEG_POINT[split, n] = {m: float(v) for m, v in point.items()}
        rows.append({"split": split, "arm": n, **{m: f"{float(point[m]):.4f} [{np.percentile(boot[m], 2.5):.3f}, "
                                                     f"{np.percentile(boot[m], 97.5):.3f}]" for m in point}})
    if len(counts) == 2:
        (na, ca), (nb, cb) = counts.items()
        pa, pb = score_parts(ca[idx].sum(axis=1)), score_parts(cb[idx].sum(axis=1))
        qa_, qb = score_parts(ca.sum(axis=0)), score_parts(cb.sum(axis=0))
        deltas += [{"split": split, "metric": m, "delta (EXP-02C - EXP-01)": float(qb[m] - qa_[m]),
                    "ci_low": float(np.percentile(pb[m] - pa[m], 2.5)), "ci_high": float(np.percentile(pb[m] - pa[m], 97.5))}
                   for m in qa_]
display(pd.DataFrame(rows))
if deltas:
    DELTA_SEG = pd.DataFrame(deltas)
    DELTA_SEG["significant (95% CI excludes 0)"] = (DELTA_SEG["ci_low"] > 0) | (DELTA_SEG["ci_high"] < 0)
    display(DELTA_SEG.round(4))
else:
    print("EXP-01 scenes not cached here (artifacts/cache_exp01/scenes_thr0.175.jsonl): EXP-01 mask comparison skipped")


def stage_ms(path, stage):
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []
    return {r["image_id"]: r for r in recs if r.get("type") == "stage" and r["stage_name"] == stage}


det_traces = {n: (CACHE_DIR / ("base" if n == "owlv2b" else "large") / "trace_detection.jsonl") if n.startswith("owlv2")
              else (GDINO_DIR / ("trace_detection.jsonl" if DRY_RUN else f"trace_detection_{GD_VARIANT}.jsonl")) for n in SEL_MEMBERS}
det = {n: stage_ms(p, "detection_exp02") for n, p in det_traces.items()}
seg = stage_ms(EXP02C_DIR / "trace_segmentation.jsonl", "segmentation")
build = stage_ms(EXP02C_DIR / "trace_segmentation.jsonl", "scene_build")
ON_GPU = all(m.get("device") == "cuda" for m in [SEG_META] + [META["mmgdino", GD_VARIANT], META["base", "chunked_raw"]])
VRAM_PEAK_MB = max(r["vram_peak_mb"] for d in [*det.values(), seg] for r in d.values()) if ON_GPU else None
LAT = pd.DataFrame([{**{f"det {n} s/img (5 views)": np.mean([r["duration_ms"] for r in det[n].values()]) / 1000 for n in det},
                     "fusion ms/img (CPU)": 1000 * FUSE_SEC,
                     "SAM2 s/img": np.mean([r["duration_ms"] for r in seg.values()]) / 1000 if seg else None,
                     "scene build s/img": np.mean([r["duration_ms"] for r in build.values()]) / 1000 if build else None,
                     "peak VRAM MB": VRAM_PEAK_MB}])
display(LAT.round(3))
'''),
    ("md", "62_box_mask"),
    ("code", r'''
QUESTIONS = [json.loads(l) for l in PATHS.questions_dev.read_text(encoding="utf-8").splitlines() if l.strip()]
questions_by_key = defaultdict(list)
for q in QUESTIONS:
    questions_by_key[q["image_file"].rsplit(".", 1)[0]].append(q)
SUBMISSION = (EXP02C_DIR / "submission_exp02_perception_dryrun.csv" if DRY_RUN
              else REPO / "submissions" / "submission_exp02_perception.csv")
SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
sample_rows = ev.read_submission(PATHS.sample_submission)
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
print(f"{SUBMISSION.relative_to(REPO)}: {len(written)} rows, {SUBMISSION.stat().st_size / 1e6:.1f} MB | "
      f"objects/img test {np.mean([len(SCENES[r['image_id']]['objects']) for r in written[:60]]):.1f}")
'''),
    ("md", "63_submission"),
    ("code", r'''
if DRY_RUN:
    infer_fn = None  # the stand-in config is not a live detector
else:
    from src.segmentation_sam2 import Sam2Segmenter
    LIVE = fu.EnsembleDetector.from_config(EXP02C_CONFIG, cfg.CATEGORIES)
    SEGMENTER = Sam2Segmenter()

    def infer_fn(image_path):
        scene, _ = pl.infer_scene(image_path, LIVE, SEGMENTER, cfg.CATEGORIES, score_threshold=0.0)
        return scene

t0 = time.perf_counter()
REPORT = ev.run_full_verification_suite(SUBMISSION, infer_fn=infer_fn, qa_fn=qa.answer_questions,
                                        vram_peak_mb=VRAM_PEAK_MB, raise_on_fail=False)
print(REPORT)
print(f"\nsuite time: {time.perf_counter() - t0:.1f}s | FAIL count: {len(REPORT.failed)}")
assert not REPORT.failed, "Submission #2 must pass the contract suite with 0 FAIL"
'''),
    ("md", "64_suite"),
    ("code", r'''
SUMMARY_C = {
    "selection": {"rule": "max CAL Box F1; groups by CAL coordinate ascent", "member_arm": SEL, "members": SEL_MEMBERS,
                  "mmgdino_variant": GD_VARIANT, "config": str(CONFIG_PATH.relative_to(REPO)), "config_sha": SHA,
                  "group_thresholds": THR, "fp_prone": sorted(cfg.CATEGORIES[c] for c in FP_PRONE), "dry_run": DRY_RUN},
    "box_f1": {f"{s}/{n}": round(F1_BY[s, n], 4) for s, n in F1_BY},
    "val_deltas_f1": {c: {k: round(float(r[k]), 4) for k in ("delta", "ci_low", "ci_high")} for c, r in s4.iterrows()},
    "box_mask": {f"{s}/{n}": {m: round(v, 4) for m, v in d.items()} for (s, n), d in SEG_POINT.items()},
    "suite": REPORT.statuses(), "vram_peak_mb": VRAM_PEAK_MB,
}
(EXP02C_DIR / "exp02c_summary.json").write_text(json.dumps(SUMMARY_C, indent=2), encoding="utf-8")
print(json.dumps(SUMMARY_C, indent=2))
'''),
    ("md", "70_conclusions"),
]
