"""Contract tests for T1 (S1): config, scene builder, tracer and LVIS evaluator."""

import json
import unittest

import numpy as np

from src import config_and_seeds as cfg
from src import lvis_evaluator as ev
from src import scene_builder as sb
from src.trace_logger import ExecutionTracer


class TestConfigAndSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = cfg.get_dataset_paths()
        cls.annotations = json.loads(cls.paths.annotations.read_text())
        cls.qa_ids = cfg.load_qa_image_ids(cls.paths.questions_dev)

    def test_paths_resolve_to_existing_files(self):
        for path in (self.paths.schema, self.paths.sample_submission, self.paths.categories,
                     self.paths.test_images, self.paths.questions_dev, self.paths.annotations):
            self.assertTrue(path.is_file(), path)

    def test_loads_120_categories(self):
        categories = cfg.load_categories(self.paths.categories)
        self.assertEqual(sorted(categories), list(range(1, 121)))
        self.assertEqual(categories[119], "fireplug")

    def test_aliases_cover_all_categories_with_official_name_first(self):
        aliases = cfg.build_prompt_aliases(cfg.load_categories(self.paths.categories))
        self.assertEqual(sorted(aliases), list(range(1, 121)))
        for cid, entry in aliases.items():
            self.assertEqual(entry["prompts"][0], entry["name"])
            self.assertEqual(len(entry["prompts"]), len(set(entry["prompts"])))
        self.assertIn("fire hydrant", aliases[119]["prompts"])
        self.assertIn("eyeglasses", aliases[90]["prompts"])
        self.assertIn("tv", aliases[47]["prompts"])
        self.assertIn("garbage can", aliases[19]["prompts"])

    def test_alias_override_with_unknown_name_fails_fast(self):
        with self.assertRaises(ValueError):
            cfg.build_prompt_aliases({1: "sofa"}, overrides={"not a category": ["x"]})

    def test_size_thresholds_boundaries(self):
        self.assertEqual(cfg.size_from_area_ratio(0.00349), "small")
        self.assertEqual(cfg.size_from_area_ratio(0.0035), "medium")
        self.assertEqual(cfg.size_from_area_ratio(0.03499), "medium")
        self.assertEqual(cfg.size_from_area_ratio(0.035), "large")

    def test_split_sizes_and_qa_balance(self):
        split = cfg.make_cal_val_split(self.annotations, self.qa_ids, seed=cfg.SEED)
        cal, val = set(split["cal"]), set(split["val"])
        self.assertEqual((len(cal), len(val)), (80, 40))
        self.assertFalse(cal & val)
        self.assertEqual(cal | val, set(range(1, 121)))
        self.assertEqual(len(cal & set(self.qa_ids)), 10)
        self.assertEqual(len(val & set(self.qa_ids)), 10)

    def test_resolve_device_prefers_env_override_then_cuda_availability(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"TPI_DEVICE": "cpu"}):
            self.assertEqual(cfg.resolve_device(), "cpu")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TPI_DEVICE", None)
            try:
                import torch
                expected = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                expected = "cpu"
            self.assertEqual(cfg.resolve_device(), expected)

    def test_split_is_deterministic(self):
        a = cfg.make_cal_val_split(self.annotations, self.qa_ids, seed=cfg.SEED)
        b = cfg.make_cal_val_split(self.annotations, self.qa_ids, seed=cfg.SEED)
        self.assertEqual(a, b)


class TestRle(unittest.TestCase):
    def test_round_trip_random_masks(self):
        rng = np.random.default_rng(cfg.SEED)
        for shape in [(1, 1), (7, 5), (64, 48)]:
            mask = rng.random(shape) > 0.5
            rle = sb.encode_rle(mask)
            self.assertEqual(rle["size"], list(shape))
            self.assertEqual(sum(rle["counts"]), shape[0] * shape[1])
            np.testing.assert_array_equal(sb.decode_rle(rle), mask)

    def test_counts_start_with_zero_run_when_first_pixel_is_set(self):
        mask = np.ones((3, 2), dtype=bool)
        self.assertEqual(sb.encode_rle(mask)["counts"], [0, 6])
        self.assertEqual(sb.encode_rle(np.zeros((3, 2), dtype=bool))["counts"], [6])

    def test_column_major_order(self):
        mask = np.zeros((2, 3), dtype=bool)
        mask[1, 0] = True  # column-major index 1
        self.assertEqual(sb.encode_rle(mask)["counts"], [1, 1, 4])

    def test_reproduces_sample_submission_row(self):
        mask = np.zeros((640, 425), dtype=bool)
        mask[0, 0] = True
        self.assertEqual(sb.encode_rle(mask), {"size": [640, 425], "counts": [0, 1, 271999]})
        geom = sb.compute_object_geometry(mask)
        self.assertEqual(geom["centroid"], [0.0, 0.0])
        self.assertAlmostEqual(geom["area_ratio"], 3.6764705882352942e-06)
        self.assertEqual((geom["position"], geom["size"]), ("top_left", "small"))

    def test_decode_rejects_inconsistent_counts(self):
        with self.assertRaises(ValueError):
            sb.decode_rle({"size": [2, 2], "counts": [1, 1]})


class TestObjectGeometry(unittest.TestCase):
    def test_empty_mask_falls_back_to_clipped_bbox(self):
        mask, used = sb.ensure_non_empty_mask(np.zeros((10, 10), dtype=bool), [-5, 2, 4, 6])
        self.assertTrue(used)
        self.assertEqual(int(mask.sum()), 4 * 4)

    def test_degenerate_bbox_still_yields_one_pixel(self):
        mask, used = sb.ensure_non_empty_mask(np.zeros((10, 10), dtype=bool), [20, 20, 20, 20])
        self.assertTrue(used)
        self.assertEqual(int(mask.sum()), 1)

    def test_position_grid_corners_and_center(self):
        cases = {(0, 0): "top_left", (99, 0): "top_right", (50, 50): "center",
                 (0, 99): "bottom_left", (99, 99): "bottom_right", (50, 99): "bottom_center"}
        for (x, y), expected in cases.items():
            mask = np.zeros((100, 100), dtype=bool)
            mask[y, x] = True
            self.assertEqual(sb.compute_object_geometry(mask)["position"], expected)

    def test_dominant_color_uses_eroded_copy_with_fallback(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[...] = (255, 0, 0)
        image[5:15, 5:15] = (0, 0, 255)
        border = np.zeros((20, 20), dtype=bool)
        border[4:16, 4:16] = True  # 1px red ring around a blue core
        self.assertEqual(sb.dominant_color(image, border)["name"], "blue")
        single = np.zeros((20, 20), dtype=bool)
        single[0, 0] = True  # erosion empties it -> fallback
        self.assertEqual(sb.dominant_color(image, single)["name"], "red")

    def test_build_object_has_required_keys_and_positive_area(self):
        obj = sb.build_object("o1", 3, 0.9, [0, 0, 4, 4], np.zeros((10, 10), dtype=bool))
        self.assertTrue(set(sb.REQUIRED_OBJECT_KEYS) <= set(obj))
        self.assertGreater(obj["area_ratio"], 0)


def _square_object(obj_id, x, y, side=4, conf=0.5, shape=(100, 100)):
    mask = np.zeros(shape, dtype=bool)
    mask[y:y + side, x:x + side] = True
    return sb.build_object(obj_id, 1, conf, [x, y, x + side, y + side], mask)


class TestRelations(unittest.TestCase):
    def test_symmetric_directional_and_overlap_relations(self):
        a, b, c = _square_object("a", 0, 0), _square_object("b", 50, 80), _square_object("c", 2, 2)
        rels = {(r["subject"], r["predicate"], r["object"]) for r in sb.build_relations([a, b, c])}
        self.assertIn(("a", "left_of", "b"), rels)
        self.assertIn(("b", "right_of", "a"), rels)
        self.assertIn(("a", "above", "b"), rels)
        self.assertIn(("b", "below", "a"), rels)
        self.assertIn(("a", "overlaps", "c"), rels)
        self.assertIn(("c", "overlaps", "a"), rels)
        self.assertNotIn(("a", "overlaps", "b"), rels)

    def test_equal_centroids_on_an_axis_emit_no_relation_on_that_axis(self):
        a, b = _square_object("a", 10, 0), _square_object("b", 10, 50)
        predicates = {r["predicate"] for r in sb.build_relations([a, b])}
        self.assertEqual(predicates, {"above", "below"})

    def test_touching_boxes_do_not_overlap(self):
        a, b = _square_object("a", 0, 0), _square_object("b", 4, 0)
        self.assertNotIn("overlaps", {r["predicate"] for r in sb.build_relations([a, b])})

    def test_dense_scene_respects_relation_cap(self):
        rng = np.random.default_rng(cfg.SEED)
        objs = [_square_object(f"o{i}", int(x), int(y), side=30, conf=float(rng.random()))
                for i, (x, y) in enumerate(rng.integers(0, 70, size=(150, 2)))]
        rels = sb.build_relations(objs)
        self.assertLess(len(rels), 20000)
        used = {r["subject"] for r in rels}
        top = {o["id"] for o in sorted(objs, key=lambda o: -o["confidence"])[:95]}
        self.assertTrue(used <= top)
        self.assertEqual(_asymmetries(rels), [])


def _asymmetries(relations):
    inverse = {"left_of": "right_of", "right_of": "left_of", "above": "below", "below": "above",
               "overlaps": "overlaps"}
    triples = {(r["subject"], r["predicate"], r["object"]) for r in relations}
    return [t for t in triples if (t[2], inverse[t[1]], t[0]) not in triples]


class TestSceneJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(cfg.get_dataset_paths().schema.read_text())

    def test_empty_scene_validates(self):
        scene = sb.build_scene_json("tpi_0001", 640, 480, [])
        self.assertEqual(sb.validate_scene(scene, self.schema), [])
        self.assertEqual((scene["objects"], scene["relations"]), ([], []))

    def test_objects_truncated_to_300_by_confidence(self):
        objs = [_square_object(f"o{i}", i % 90, (i * 7) % 90, conf=i / 400) for i in range(350)]
        scene = sb.build_scene_json("x", 100, 100, objs)
        self.assertEqual(len(scene["objects"]), 300)
        self.assertEqual(min(o["confidence"] for o in scene["objects"]), 50 / 400)
        self.assertEqual(sb.validate_scene(scene, self.schema), [])

    def test_invalid_area_ratio_is_reported(self):
        obj = _square_object("a", 0, 0)
        obj["area_ratio"] = 0.0
        scene = sb.build_scene_json("x", 100, 100, [obj])
        self.assertTrue(sb.validate_scene(scene, self.schema))


class TestExecutionTracer(unittest.TestCase):
    def test_stage_records_all_fields_and_object_lineage(self):
        tracer = ExecutionTracer()
        with tracer.stage("tpi_0001", "detection", parameters={"thr": 0.1}) as rec:
            rec.set_counts(input_count=1, output_count=3)
            rec.warn("low confidence")
        tracer.trace_object("tpi_0001", "obj_1", detector_source="owlv2", raw_score=0.4)
        record = tracer.to_records()[0]
        for key in ("image_id", "stage_name", "duration_ms", "vram_peak_mb", "input_count",
                    "output_count", "parameters_used", "warnings"):
            self.assertIn(key, record)
        self.assertEqual((record["output_count"], record["warnings"]), (3, ["low confidence"]))
        self.assertGreaterEqual(record["duration_ms"], 0)
        self.assertEqual(tracer.object_lineage("tpi_0001", "obj_1")["detector_source"], "owlv2")
        self.assertEqual(tracer.summary()["tpi_0001"]["detection"]["calls"], 1)

    def test_stage_records_failure_and_reraises(self):
        tracer = ExecutionTracer()
        with self.assertRaises(RuntimeError):
            with tracer.stage("img", "segmentation"):
                raise RuntimeError("boom")
        self.assertTrue(any("RuntimeError" in w for w in tracer.to_records()[0]["warnings"]))


def _gt_image(boxes, neg=(), not_exhaustive=(), shape=(100, 100)):
    """Synthetic GT entry: boxes is a list of (category_id, xyxy)."""
    masks = []
    for _, (x1, y1, x2, y2) in boxes:
        m = np.zeros(shape, dtype=bool)
        m[y1:y2, x1:x2] = True
        masks.append(m)
    return {"height": shape[0], "width": shape[1], "categories": [c for c, _ in boxes],
            "boxes": [list(map(float, b)) for _, b in boxes], "masks": masks,
            "pos": {c for c, _ in boxes}, "neg": set(neg), "not_exhaustive": set(not_exhaustive)}


def _pred(obj_id, cid, xyxy, conf=0.9, shape=(100, 100)):
    x1, y1, x2, y2 = xyxy
    m = np.zeros(shape, dtype=bool)
    m[y1:y2, x1:x2] = True
    return sb.build_object(obj_id, cid, conf, list(xyxy), m)


class TestFederatedEvaluator(unittest.TestCase):
    def test_policy_tp_fp_fn_ignored_and_coverage(self):
        gt = {"img": _gt_image([(1, (0, 0, 20, 20)), (1, (50, 50, 70, 70)), (2, (0, 50, 20, 70))],
                               neg=[3], not_exhaustive=[2])}
        preds = {"img": [
            _pred("tp", 1, (0, 0, 20, 20)),          # TP (cat 1)
            _pred("fp_pos", 1, (80, 0, 99, 20)),     # FP: unmatched on exhaustive positive
            _pred("tp2", 2, (0, 50, 20, 70)),        # TP (cat 2)
            _pred("ign_ne", 2, (80, 80, 99, 99)),    # ignored: not exhaustive
            _pred("fp_neg", 3, (30, 30, 40, 40)),    # FP: verified absent
            _pred("ign_unv", 4, (30, 30, 40, 40)),   # ignored: unverified
        ]}
        report = ev.evaluate_federated(preds, gt)
        for kind in ("box", "mask"):
            self.assertEqual((report[kind]["tp"], report[kind]["fp"], report[kind]["fn"]), (2, 2, 1))
            self.assertAlmostEqual(report[kind]["precision"], 0.5)
            self.assertAlmostEqual(report[kind]["recall"], 2 / 3)
        self.assertAlmostEqual(report["coverage"], 5 / 6)

    def test_iou_threshold_and_one_to_one_matching(self):
        gt = {"img": _gt_image([(1, (0, 0, 10, 10))])}
        preds = {"img": [_pred("a", 1, (0, 0, 10, 10), conf=0.9), _pred("b", 1, (0, 0, 10, 10), conf=0.8),
                         _pred("c", 1, (0, 0, 10, 4), conf=0.95)]}  # IoU 0.4 -> FP
        report = ev.evaluate_federated(preds, gt)
        self.assertEqual((report["box"]["tp"], report["box"]["fp"]), (1, 2))

    def test_gt_annotations_convert_xywh_to_xyxy(self):
        annotations = {"images": [{"id": 1, "file_name": "tpi_0001.jpg", "width": 10, "height": 10,
                                   "neg_category_ids": [5], "not_exhaustive_category_ids": []}],
                       "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [1, 2, 3, 4],
                                        "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6]]}]}
        gt = ev.load_gt(annotations)["tpi_0001"]
        self.assertEqual(gt["boxes"], [[1.0, 2.0, 4.0, 6.0]])
        self.assertEqual((gt["pos"], gt["neg"]), ({2}, {5}))
        self.assertTrue(gt["masks"][0].any())


class TestVerificationSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = cfg.get_dataset_paths()

    def test_sample_submission_passes_structural_suite(self):
        report = ev.run_full_verification_suite(self.paths.sample_submission, require_qa=False)
        statuses = report.statuses()
        self.assertEqual(statuses["test_1_blind_inference"], "SKIPPED")
        self.assertEqual(statuses["test_8_vram_budget"], "SKIPPED")
        self.assertNotIn("FAIL", statuses.values())

    def test_sample_submission_fails_strict_qa(self):
        report = ev.run_full_verification_suite(self.paths.sample_submission, require_qa=True,
                                                raise_on_fail=False)
        self.assertEqual(report.statuses()["test_4_qa_integrity"], "FAIL")

    def test_corrupted_area_ratio_fails(self):
        import csv
        import tempfile
        from pathlib import Path

        rows = ev.read_submission(self.paths.sample_submission)
        scene = json.loads(rows[0]["SceneJSON"])
        scene["objects"][0]["area_ratio"] = 0.0
        rows[0]["SceneJSON"] = json.dumps(scene)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image_id", "SceneJSON"])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(AssertionError):
                ev.run_full_verification_suite(path, require_qa=False)


if __name__ == "__main__":
    unittest.main()
