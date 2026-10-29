"""Model-free tests for the EXP-02C segmentation runner (frozen ensemble boxes -> scenes cache)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import run_exp02c_segmentation as seg
from src import config_and_seeds as cfg
from src import detection_fusion as fu
from src import pipeline as pl
from src import scene_builder as sb
from src.perception_detectors import Detection


class _ExplodingSegmenter:
    def __init__(self, name="explode"):
        self.name = name

    def segment(self, image, boxes):
        raise AssertionError("already segmented images must be skipped")


class TestSegmentScenes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = []
        for key in ("tpi_0001", "test_0001"):
            path = self.root / f"{key}.jpg"
            Image.fromarray(np.full((48, 64, 3), 90, dtype=np.uint8)).save(path)
            self.paths.append(path)
        self.dets = {"tpi_0001": [Detection(1, 0.8, (2.0, 3.0, 20.0, 30.0), "wbf:owlv2", raw_score=0.3)],
                     "test_0001": []}

    def tearDown(self):
        self.tmp.cleanup()

    def test_box_fill_masks_are_the_box_region(self):
        image = Image.open(self.paths[0])
        (mask, score), = seg.BoxFillSegmenter().segment(image, [(2.0, 3.0, 20.0, 30.0)])
        self.assertEqual((mask.shape, int(mask.sum()), score), ((48, 64), 18 * 27, 0.0))

    def test_scenes_are_valid_and_the_run_resumes(self):
        out = self.root / "exp02c"
        seg.segment_scenes(self.dets, self.paths, seg.BoxFillSegmenter(), out, config_sha="abc")
        scenes = [json.loads(l) for l in (out / "scenes.jsonl").read_text().splitlines()]
        schema = json.loads(cfg.get_dataset_paths().schema.read_text())
        self.assertEqual([s["image"]["id"] for s in scenes], ["tpi_0001", "test_0001"])
        self.assertTrue(all(sb.validate_scene(s, schema) == [] for s in scenes))
        self.assertEqual(scenes[0]["objects"][0]["detector_source"], "wbf:owlv2")
        seg.segment_scenes(self.dets, self.paths, _ExplodingSegmenter("box_fill"), out, config_sha="abc")  # resumes
        self.assertEqual(len((out / "scenes.jsonl").read_text().splitlines()), 2)
        self.assertEqual(json.loads((out / "scenes.meta.json").read_text())["config_sha"], "abc")

    def test_scenes_from_another_config_fail_fast(self):
        out = self.root / "exp02c"
        seg.segment_scenes(self.dets, self.paths, seg.BoxFillSegmenter(), out, config_sha="abc")
        with self.assertRaises(ValueError):
            seg.segment_scenes(self.dets, self.paths, seg.BoxFillSegmenter(), out, config_sha="other")
        with self.assertRaises(ValueError):  # box-fill dry-run scenes never mix with SAM2 scenes
            seg.segment_scenes(self.dets, self.paths, _ExplodingSegmenter(), out, config_sha="abc")

    def test_member_caches_are_loaded_retagged_and_fused(self):
        cache = pl.save_detection_cache({"tpi_0001": [Detection(1, 0.6, (2.0, 3.0, 20.0, 30.0), "owlv2:global")]},
                                        self.root / "d.json.gz", {"image_sizes": {"tpi_0001": [64, 48]}})
        ident = fu.QuantileCalibrator((0.0, 0.5, 1.0), (0.0, 0.5, 1.0))
        config = fu.EnsembleConfig(members={"owlv2b": fu.MemberSpec("owlv2", "m", "chunked_raw", cache="d.json.gz")},
                                   calibrators={"owlv2b:global": ident, "owlv2b:tile": ident},
                                   tile_category_ids=frozenset(), tile_max_area_ratio=None, group_of={},
                                   thresholds={g: 0.5 for g in fu.GROUPS})
        self.assertTrue(cache.exists())
        fused, sizes = seg.fused_detections(config, self.root)
        self.assertEqual(sizes, {"tpi_0001": [64, 48]})
        self.assertEqual([(d.category_id, d.score, d.source) for d in fused["tpi_0001"]], [(1, 0.6, "wbf:owlv2b")])


class TestNotebookMarkdownRefresh(unittest.TestCase):
    def test_refresh_replaces_markdown_and_keeps_code_outputs(self):
        import nbformat

        from scripts import build_notebook_02 as b02

        nb = b02.build()
        code = next(c for c in nb.cells if c.cell_type == "code")
        code.outputs = [nbformat.v4.new_output("stream", name="stdout", text="kept\n")]
        first_md = next(c for c in nb.cells if c.cell_type == "markdown")
        first_md.source = "stale"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nb.ipynb"
            nbformat.write(nb, path)
            out = b02.refresh_markdown(path)
        self.assertEqual(next(c for c in out.cells if c.cell_type == "code").outputs[0]["text"], "kept\n")
        self.assertEqual(next(c for c in out.cells if c.cell_type == "markdown").source,
                         b02.md(first_md.metadata["analysis"]))
        self.assertEqual(len(out.cells), len(nb.cells))

    def test_refresh_rejects_a_notebook_with_another_layout(self):
        import nbformat

        from scripts import build_notebook_02 as b02

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nb.ipynb"
            nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("x")]), path)
            with self.assertRaises(ValueError):
                b02.refresh_markdown(path)


if __name__ == "__main__":
    unittest.main()
