"""Model-free tests for T2 (S2): detector post-processing, mask guarantee and the inference pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src import config_and_seeds as cfg
from src import lvis_evaluator as ev
from src import pipeline as pl
from src import scene_builder as sb
from src.perception_detectors import Detection, clip_box, filter_detections, per_class_nms
from src.segmentation_sam2 import guarantee_non_empty


class TestDetectorPostprocessing(unittest.TestCase):
    def test_clip_box_to_image_and_drop_degenerate(self):
        # OWLv2 predicts on the padded square, so boxes may extend past the real image.
        self.assertEqual(clip_box((-5, 10, 700, 50), width=640, height=480), (0.0, 10.0, 640.0, 50.0))
        self.assertIsNone(clip_box((650, 10, 700, 50), width=640, height=480))
        self.assertIsNone(clip_box((10, 500, 50, 600), width=640, height=480))

    def test_per_class_nms_keeps_best_and_other_classes(self):
        dets = [Detection(1, 0.9, (0, 0, 10, 10)), Detection(1, 0.8, (1, 1, 11, 11)),
                Detection(2, 0.7, (1, 1, 11, 11)), Detection(1, 0.6, (50, 50, 60, 60))]
        kept = per_class_nms(dets, iou_threshold=0.5)
        self.assertEqual([(d.category_id, d.score) for d in kept], [(1, 0.9), (2, 0.7), (1, 0.6)])

    def test_filter_detections_threshold_and_cap(self):
        dets = [Detection(1, s / 100, (s, 0, s + 5, 5)) for s in range(1, 100)]
        kept = filter_detections(dets, score_threshold=0.5, max_detections=10)
        self.assertEqual(len(kept), 10)
        self.assertTrue(all(d.score >= 0.5 for d in kept))
        self.assertEqual(kept[0].score, 0.99)


class TestMaskGuarantee(unittest.TestCase):
    def test_empty_mask_falls_back_to_box_and_is_flagged(self):
        mask, fallback = guarantee_non_empty(np.zeros((20, 30), dtype=bool), (2, 3, 8, 9))
        self.assertTrue(fallback)
        self.assertEqual(int(mask.sum()), 36)

    def test_non_empty_mask_is_untouched(self):
        m = np.zeros((20, 30), dtype=bool)
        m[5, 5] = True
        mask, fallback = guarantee_non_empty(m, (0, 0, 10, 10))
        self.assertFalse(fallback)
        self.assertEqual(int(mask.sum()), 1)


class _FakeDetector:
    name = "fake"

    def detect(self, image):
        w, h = image.size
        return [Detection(17, 0.9, (5, 5, w / 2, h / 2)), Detection(13, 0.3, (w / 2, h / 2, w - 1, h - 1)),
                Detection(12, 0.05, (0, 0, 4, 4))]


class _FakeSegmenter:
    name = "fake"

    def segment(self, image, boxes):
        w, h = image.size
        out = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            m = np.zeros((h, w), dtype=bool)
            if i == 0:  # the second box gets an empty mask to exercise the guarantee
                m[int(y1):int(y2), int(x1):int(x2)] = True
            out.append((m, 0.8))
        return out


class TestInferScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(cfg.get_dataset_paths().schema.read_text())

    def test_infer_scene_builds_valid_scene_with_threshold_and_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "external_image.jpg"
            Image.fromarray(np.full((60, 80, 3), 128, dtype=np.uint8)).save(path)
            scene, info = pl.infer_scene(path, _FakeDetector(), _FakeSegmenter(), cfg.CATEGORIES, score_threshold=0.1)
        self.assertEqual(sb.validate_scene(scene, self.schema), [])
        self.assertEqual(scene["image"], {"id": "external_image", "width": 80, "height": 60})
        self.assertEqual([o["category_id"] for o in scene["objects"]], [17, 13])
        self.assertEqual(scene["objects"][1].get("mask_fallback"), "bbox")
        self.assertTrue(all(o["area_ratio"] > 0 for o in scene["objects"]))
        self.assertEqual(info["n_detections_raw"], 3)
        self.assertEqual(info["n_mask_fallbacks"], 1)

    def test_scene_from_cached_detections_matches_contract(self):
        image = Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8))
        dets = [Detection(1, 0.7, (0, 0, 20, 20))]
        masks = _FakeSegmenter().segment(image, [d.bbox for d in dets])
        scene, n_fallback = pl.build_scene("img", image, dets, masks, cfg.CATEGORIES)
        self.assertEqual((n_fallback, len(scene["objects"])), (0, 1))
        self.assertEqual(scene["objects"][0]["detector_source"], "owlv2")
        self.assertEqual(sb.validate_scene(scene, self.schema), [])

    def test_detection_cache_round_trip(self):
        dets = {"tpi_0001": [Detection(3, 0.25, (1.5, 2.0, 30.0, 40.0))], "test_0001": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = pl.save_detection_cache(dets, Path(tmp) / "dets.json", meta={"model": "x"})
            loaded, meta = pl.load_detection_cache(path)
        self.assertEqual(loaded, dets)
        self.assertEqual(meta["model"], "x")


class TestBoxOnlyEvaluation(unittest.TestCase):
    def test_box_only_mode_ignores_missing_masks(self):
        gt = {"img": {"height": 100, "width": 100, "categories": [1], "boxes": [[10, 10, 50, 50]],
                      "masks": [np.ones((100, 100), dtype=bool)], "pos": {1}, "neg": set(), "not_exhaustive": set()}}
        preds = {"img": [{"category_id": 1, "confidence": 0.9, "bbox": [10, 10, 50, 50]}]}
        report = ev.evaluate_federated(preds, gt, with_masks=False)
        self.assertEqual(report["box"]["f1"], 1.0)
        self.assertIsNone(report["mask"])


if __name__ == "__main__":
    unittest.main()
