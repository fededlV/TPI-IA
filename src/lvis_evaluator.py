"""Exact LVIS federated evaluator (boxes and RLE masks, IoU >= 0.5) and the 8-test contract suite."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src import scene_builder as sb
from src.config_and_seeds import get_dataset_paths, load_qa_image_ids, size_from_area_ratio

IOU_THRESHOLD = 0.5
VRAM_BUDGET_MB = 14 * 1024
EMPTY_EVIDENCE_ANSWERS = (0, "no", "unsupported", "insufficient_information")
INVERSE_PREDICATE = {"left_of": "right_of", "right_of": "left_of", "above": "below", "below": "above",
                     "overlaps": "overlaps"}

csv.field_size_limit(2 ** 31 - 1)


# --------------------------------------------------------------------------- ground truth


def image_key(file_name: str) -> str:
    return Path(file_name).stem


def load_gt(annotations: dict) -> dict[str, dict]:
    """Per image key ('tpi_0001'): xyxy boxes, rasterized masks and pos/neg/not_exhaustive sets."""
    gt: dict[str, dict] = {}
    by_id = {}
    for img in annotations["images"]:
        entry = {"height": img["height"], "width": img["width"], "categories": [], "boxes": [], "masks": [],
                 "pos": set(), "neg": set(img.get("neg_category_ids", [])),
                 "not_exhaustive": set(img.get("not_exhaustive_category_ids", []))}
        gt[image_key(img["file_name"])] = entry
        by_id[img["id"]] = entry
    for ann in annotations["annotations"]:
        entry = by_id[ann["image_id"]]
        box = sb.xywh_to_xyxy(ann["bbox"])
        mask = sb.polygon_to_mask(ann["segmentation"], entry["height"], entry["width"])
        mask, _ = sb.ensure_non_empty_mask(mask, box)
        entry["categories"].append(ann["category_id"])
        entry["boxes"].append(box)
        entry["masks"].append(mask)
        entry["pos"].add(ann["category_id"])
    return gt


# --------------------------------------------------------------------------- matching


def box_iou(a: list[float], b: list[float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _greedy_match(preds: list[dict], gts: list[Any], iou_fn: Callable, iou_thr: float) -> tuple[int, int]:
    """One-to-one greedy matching in descending confidence; returns (matched, unmatched_preds)."""
    taken = [False] * len(gts)
    matched = 0
    for p in sorted(preds, key=lambda o: -o["confidence"]):
        best, best_iou = -1, iou_thr
        for g_idx, g in enumerate(gts):
            if taken[g_idx]:
                continue
            iou = iou_fn(p, g)
            if iou >= best_iou:
                best, best_iou = g_idx, iou
        if best >= 0:
            taken[best] = True
            matched += 1
    return matched, len(preds) - matched


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_federated(predictions_by_image: dict[str, list[dict]], gt: dict[str, dict],
                       iou_thr: float = IOU_THRESHOLD) -> dict:
    """LVIS federated evaluation over the images present in both predictions and GT."""
    counts = {"box": [0, 0, 0], "mask": [0, 0, 0]}
    n_total = n_evaluated = 0
    images = [k for k in predictions_by_image if k in gt]
    for key in images:
        g, preds = gt[key], predictions_by_image[key]
        n_total += len(preds)
        mask_cache = {id(p): sb.decode_rle(p["mask"]) for p in preds}
        for cid in sorted({p["category_id"] for p in preds} | g["pos"]):
            cat_preds = [p for p in preds if p["category_id"] == cid]
            if cid in g["pos"]:
                n_evaluated += len(cat_preds)
                idx = [i for i, c in enumerate(g["categories"]) if c == cid]
                for kind, gts, fn in (
                    ("box", [g["boxes"][i] for i in idx], lambda p, b: box_iou(p["bbox"], b)),
                    ("mask", [g["masks"][i] for i in idx], lambda p, m: mask_iou(mask_cache[id(p)], m)),
                ):
                    tp, unmatched = _greedy_match(cat_preds, gts, fn, iou_thr)
                    counts[kind][0] += tp
                    counts[kind][1] += 0 if cid in g["not_exhaustive"] else unmatched
                    counts[kind][2] += len(gts) - tp
            elif cid in g["neg"]:
                n_evaluated += len(cat_preds)
                for kind in counts:
                    counts[kind][1] += len(cat_preds)
    return {"images": len(images), "box": _prf(*counts["box"]), "mask": _prf(*counts["mask"]),
            "coverage": n_evaluated / n_total if n_total else 0.0,
            "predictions_total": n_total, "predictions_evaluated": n_evaluated}


def format_report(report: dict) -> str:
    b, m = report["box"], report["mask"]
    return (f"Box  P={b['precision']:.4f} R={b['recall']:.4f} F1={b['f1']:.4f} (TP={b['tp']} FP={b['fp']} FN={b['fn']})\n"
            f"Mask P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} (TP={m['tp']} FP={m['fp']} FN={m['fn']})\n"
            f"LVIS Evaluation Coverage={report['coverage']:.4f} over {report['images']} images")


# --------------------------------------------------------------------------- verification suite


@dataclass
class TestResult:
    name: str
    status: str  # PASS | PARTIAL | SKIPPED | FAIL
    details: list[str] = field(default_factory=list)


@dataclass
class SuiteReport:
    results: list[TestResult] = field(default_factory=list)

    def statuses(self) -> dict[str, str]:
        return {r.name: r.status for r in self.results}

    @property
    def failed(self) -> list[TestResult]:
        return [r for r in self.results if r.status == "FAIL"]

    def __str__(self) -> str:
        lines = []
        for r in self.results:
            lines.append(f"[{r.status:7}] {r.name}")
            lines.extend(f"          - {d}" for d in r.details[:10])
        return "\n".join(lines)


def read_submission(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_submission_header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f), [])


def _result(name: str, errors: list[str], partial: list[str] | None = None) -> TestResult:
    if errors:
        return TestResult(name, "FAIL", errors)
    return TestResult(name, "PARTIAL" if partial else "PASS", partial or [])


def _test_blind_inference(infer_fn, schema, annotations_path: Path) -> TestResult:
    name = "test_1_blind_inference"
    if infer_fn is None:
        return TestResult(name, "SKIPPED", ["no infer_fn hook provided (available from T2)"])
    from PIL import Image

    touched: list[str] = []
    guard = {"active": True}
    target = annotations_path.name

    def audit(event, args):
        if guard["active"] and event == "open" and args and target in str(args[0]):
            touched.append(str(args[0]))

    sys.addaudithook(audit)
    errors = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "synthetic_0001.jpg"
            rng = np.random.default_rng(0)
            Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)).save(img_path)
            scene = infer_fn(img_path)
        errors += [f"schema: {e}" for e in sb.validate_scene(scene, schema)]
    except Exception as exc:  # report, never swallow silently
        errors.append(f"infer_fn raised {type(exc).__name__}: {exc}")
    finally:
        guard["active"] = False
    errors += [f"ground truth accessed during inference: {p}" for p in touched]
    return _result(name, errors)


def _test_edge_cases(schema, qa_fn) -> TestResult:
    errors: list[str] = []
    empty = sb.build_scene_json("edge_empty", 640, 480, [])
    errors += [f"empty scene: {e}" for e in sb.validate_scene(empty, schema)]
    rng = np.random.default_rng(0)
    dense = []
    for i in range(350):
        x, y = (int(v) for v in rng.integers(0, 600, 2))
        m = np.zeros((480, 640), dtype=bool)
        m[min(y, 470):min(y, 470) + 10, x:x + 10] = True
        dense.append(sb.build_object(f"d{i}", 1 + i % 120, float(rng.random()), [x, min(y, 470), x + 10, min(y, 470) + 10], m))
    start = time.perf_counter()
    scene = sb.build_scene_json("edge_dense", 640, 480, dense)
    elapsed = time.perf_counter() - start
    if len(scene["objects"]) > sb.MAX_OBJECTS or len(scene["relations"]) >= sb.MAX_RELATIONS:
        errors.append(f"dense scene caps violated: {len(scene['objects'])} objects, {len(scene['relations'])} relations")
    if elapsed >= 2.0:
        errors.append(f"dense scene took {elapsed:.2f}s (>= 2s)")
    errors += [f"dense scene: {e}" for e in sb.validate_scene(scene, schema)[:5]]
    if qa_fn is None:
        return _result("test_2_edge_cases", errors, ["QA sub-check SKIPPED: no qa_fn hook provided (available from T2)"])
    expected = {"count": 0, "existence": "no", "position": "insufficient_information", "size": "insufficient_information"}
    questions = [{"question_id": f"EDGE-{t}", "type": t, "question": t, "class_en": ["person"]}
                 for t in (*expected, "description")]
    try:
        answers = {a["question_id"]: a for a in qa_fn(empty, questions)}
        for t, value in expected.items():
            a = answers.get(f"EDGE-{t}", {})
            if a.get("normalized_answer") != value or type(a.get("normalized_answer")) is not type(value) or a.get("evidence"):
                errors.append(f"empty-scene {t}: got {a.get('normalized_answer')!r} evidence={a.get('evidence')}")
        if not answers.get("EDGE-description", {}).get("answer"):
            errors.append("empty-scene description is empty")
    except Exception as exc:
        errors.append(f"qa_fn raised {type(exc).__name__}: {exc}")
    return _result("test_2_edge_cases", errors)


def _test_symmetry(scenes: dict[str, dict]) -> TestResult:
    errors = []
    for key, scene in scenes.items():
        triples = {(r["subject"], r["predicate"], r["object"]) for r in scene.get("relations", [])
                   if r.get("predicate") in INVERSE_PREDICATE}
        missing = [t for t in triples if (t[2], INVERSE_PREDICATE[t[1]], t[0]) not in triples]
        ids = {o["id"] for o in scene.get("objects", [])}
        dangling = [t for t in triples if t[0] not in ids or t[2] not in ids]
        errors += [f"{key}: missing inverse of {t}" for t in missing[:3]]
        errors += [f"{key}: relation references unknown object {t}" for t in dangling[:3]]
    return _result("test_3_relation_symmetry", errors)


def _test_qa_integrity(scenes: dict[str, dict], expected_qids: dict[str, list[str]], require_qa: bool) -> TestResult:
    errors = []
    for key, scene in scenes.items():
        ids = {o["id"] for o in scene.get("objects", [])}
        for field_name in ("questions", "questions_llm"):
            entries = scene.get(field_name)
            if require_qa and key in expected_qids:
                got = sorted(q.get("question_id") for q in entries or [])
                if got != sorted(expected_qids[key]):
                    errors.append(f"{key}.{field_name}: expected {sorted(expected_qids[key])}, got {got}")
            for q in entries or []:
                na, ev = q.get("normalized_answer"), q.get("evidence", [])
                if q.get("type") == "count" and na != "unsupported" and (type(na) is not int):
                    errors.append(f"{key}.{q.get('question_id')}: count normalized_answer must be int, got {na!r}")
                if len(ev) != len(set(ev)):
                    errors.append(f"{key}.{q.get('question_id')}: duplicated evidence")
                if set(ev) - ids:
                    errors.append(f"{key}.{q.get('question_id')}: evidence not in objects {sorted(set(ev) - ids)}")
                if any(na == v and type(na) is type(v) for v in EMPTY_EVIDENCE_ANSWERS) and ev:
                    errors.append(f"{key}.{q.get('question_id')}: evidence must be [] for answer {na!r}")
    return _result("test_4_qa_integrity", errors)


def _test_schema(scenes: dict[str, dict], schema) -> TestResult:
    errors = []
    for key, scene in scenes.items():
        errors += [f"{key}: {e}" for e in sb.validate_scene(scene, schema)[:3]]
    return _result("test_5_schema_compliance", errors)


def _test_masks(scenes: dict[str, dict]) -> TestResult:
    errors = []
    for key, scene in scenes.items():
        w, h = scene["image"]["width"], scene["image"]["height"]
        for o in scene.get("objects", []):
            tag = f"{key}/{o.get('id')}"
            try:
                rle = o["mask"]
                if list(rle["size"]) != [h, w]:
                    errors.append(f"{tag}: mask size {rle['size']} != [H, W]=[{h}, {w}]")
                    continue
                if len(rle["counts"]) < 1 or sum(rle["counts"]) != h * w:
                    errors.append(f"{tag}: counts must be non-empty and sum to H*W")
                    continue
                if not 0 < o["area_ratio"] <= 1:
                    errors.append(f"{tag}: area_ratio {o['area_ratio']} outside (0, 1]")
                mask = sb.decode_rle(rle)
                if not mask.any():
                    errors.append(f"{tag}: decoded mask is empty")
                    continue
                geom = sb.compute_object_geometry(mask)
                if abs(geom["area_ratio"] - o["area_ratio"]) > 1e-9:
                    errors.append(f"{tag}: area_ratio {o['area_ratio']} != mask-derived {geom['area_ratio']}")
                if max(abs(a - b) for a, b in zip(geom["centroid"], o["centroid"])) > 1e-3:
                    errors.append(f"{tag}: centroid {o['centroid']} != mask-derived {geom['centroid']}")
                if o["position"] != geom["position"] or o["size"] != size_from_area_ratio(o["area_ratio"]):
                    errors.append(f"{tag}: position/size inconsistent with mask geometry")
                x1, y1, x2, y2 = o["bbox"]
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    errors.append(f"{tag}: bbox {o['bbox']} not a valid xyxy box inside the image")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{tag}: {type(exc).__name__}: {exc}")
    return _result("test_6_rle_mask_integrity", errors)


def _test_csv_structure(header: list[str], rows: list[dict], sample_rows: list[dict], scenes: dict,
                        parse_errors: list[str]) -> TestResult:
    errors = list(parse_errors)
    if header != ["image_id", "SceneJSON"]:
        errors.append(f"columns must be exactly image_id,SceneJSON; got {header}")
    ids = [r.get("image_id") for r in rows]
    expected = [r["image_id"] for r in sample_rows]
    if ids != expected:
        errors.append(f"{len(ids)} rows / order differs from sample_submission.csv ({len(expected)} rows)")
    sample_dims = {r["image_id"]: json.loads(r["SceneJSON"])["image"] for r in sample_rows}
    for key, scene in scenes.items():
        img = scene.get("image", {})
        if img.get("id") != key:
            errors.append(f"{key}: SceneJSON image.id is {img.get('id')!r}")
        ref = sample_dims.get(key)
        if ref and (img.get("width"), img.get("height")) != (ref["width"], ref["height"]):
            errors.append(f"{key}: image size {img.get('width')}x{img.get('height')} != {ref['width']}x{ref['height']}")
    return _result("test_7_csv_structure", errors)


def _test_vram(vram_peak_mb: float | None) -> TestResult:
    name = "test_8_vram_budget"
    if vram_peak_mb is None:
        return TestResult(name, "SKIPPED", ["no recorded VRAM peak provided (available once models run)"])
    ok = vram_peak_mb < VRAM_BUDGET_MB
    return TestResult(name, "PASS" if ok else "FAIL", [f"peak {vram_peak_mb:.0f} MB vs budget {VRAM_BUDGET_MB} MB"])


def run_full_verification_suite(submission_path, *, require_qa: bool = True, infer_fn=None, qa_fn=None,
                                vram_peak_mb: float | None = None, sample_path=None, schema_path=None,
                                raise_on_fail: bool = True, verbose: bool = False) -> SuiteReport:
    """Run the 8 contract tests; raise AssertionError on any FAIL unless raise_on_fail=False."""
    paths = get_dataset_paths()
    schema = json.loads(Path(schema_path or paths.schema).read_text(encoding="utf-8"))
    rows = read_submission(Path(submission_path))
    sample_rows = read_submission(Path(sample_path or paths.sample_submission))
    scenes, parse_errors = {}, []
    for r in rows:
        try:
            scenes[r["image_id"]] = json.loads(r["SceneJSON"])
        except (TypeError, json.JSONDecodeError) as exc:
            parse_errors.append(f"{r.get('image_id')}: invalid SceneJSON ({exc})")
    expected_qids: dict[str, list[str]] = {}
    with open(paths.questions_dev, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                expected_qids.setdefault(image_key(q["image_file"]), []).append(q["question_id"])
    assert set(expected_qids) == {f"tpi_{i:04d}" for i in load_qa_image_ids(paths.questions_dev)}

    report = SuiteReport([
        _test_blind_inference(infer_fn, schema, paths.annotations),
        _test_edge_cases(schema, qa_fn),
        _test_symmetry(scenes),
        _test_qa_integrity(scenes, expected_qids, require_qa),
        _test_schema(scenes, schema),
        _test_masks(scenes),
        _test_csv_structure(read_submission_header(Path(submission_path)), rows, sample_rows, scenes, parse_errors),
        _test_vram(vram_peak_mb),
    ])
    if verbose:
        print(report)
    if raise_on_fail and report.failed:
        raise AssertionError(f"Verification suite failed:\n{report}")
    return report


def validate_submission_df(submission_path, **kwargs) -> SuiteReport:
    """Alias kept for the master-plan prompt naming (structural tests; hooks optional)."""
    return run_full_verification_suite(submission_path, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 8-test submission verification suite.")
    parser.add_argument("submission")
    parser.add_argument("--no-require-qa", action="store_true")
    args = parser.parse_args(argv)
    report = run_full_verification_suite(args.submission, require_qa=not args.no_require_qa, raise_on_fail=False)
    print(report)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
