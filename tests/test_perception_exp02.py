"""Model-free tests for T3 (S3): chunked prompting, LVIS alias remapping, sibling rules and 2x2 tiling."""

import unittest

import numpy as np

from src import config_and_seeds as cfg
from src import perception_detectors as pdet
from src.perception_detectors import Detection

CAT_ID = {name: cid for cid, name in cfg.CATEGORIES.items()}


class TestAliasTable(unittest.TestCase):
    def test_every_prompt_maps_to_exactly_one_category(self):
        owner = {}
        for cid, entry in cfg.LVIS_PROMPT_ALIASES.items():
            self.assertEqual(entry["prompts"][0], entry["name"])
            for prompt in entry["prompts"]:
                self.assertNotIn(prompt, owner, f"{prompt!r} maps to {owner.get(prompt)} and {cid}")
                owner[prompt] = cid

    def test_duplicate_prompt_across_categories_fails_fast(self):
        with self.assertRaises(ValueError):
            cfg.build_prompt_aliases({11: "handle", 81: "doorknob"},
                                     overrides={"handle": ["door handle"], "doorknob": ["door handle"]})

    def test_archaic_names_have_visual_aliases(self):
        self.assertIn("fire hydrant", cfg.LVIS_PROMPT_ALIASES[119]["prompts"])
        self.assertIn("eyeglasses", cfg.LVIS_PROMPT_ALIASES[90]["prompts"])

    def test_query_table_remaps_aliases_to_official_ids(self):
        table = pdet.build_query_table(cfg.LVIS_PROMPT_ALIASES)
        self.assertEqual(len(table.prompts), len(table.category_ids))
        self.assertEqual(len(set(table.prompts)), len(table.prompts))
        self.assertEqual(set(table.category_ids), set(range(1, 121)))
        self.assertEqual(sum(table.is_official), 120)
        i = table.prompts.index(pdet.PROMPT_TEMPLATE.format("fire hydrant"))
        self.assertEqual((table.category_ids[i], table.is_official[i]), (119, False))


class TestSemanticBlocks(unittest.TestCase):
    def setUp(self):
        self.blocks = pdet.build_blocks(cfg.CATEGORIES)

    def test_five_blocks_of_24_partition_the_vocabulary(self):
        self.assertEqual([len(b) for b in self.blocks.values()], [24] * 5)
        flat = [cid for b in self.blocks.values() for cid in b]
        self.assertEqual(sorted(flat), list(range(1, 121)))

    def test_each_sibling_group_lives_in_one_block(self):
        groups = pdet.sibling_groups(cfg.CATEGORIES)
        self.assertEqual(len(groups), len(pdet.SIBLING_GROUPS))
        for group in groups:
            homes = {name for name, ids in self.blocks.items() if group & set(ids)}
            self.assertEqual(len(homes), 1, f"siblings {group} split across {homes}")

    def test_unknown_block_member_fails_fast(self):
        with self.assertRaises(ValueError):
            pdet.build_blocks({1: "sofa"}, {"home": ("sofa", "spaceship")})


class TestVariantExtraction(unittest.TestCase):
    """Synthetic query-score matrices: 3 boxes x queries over a 4-category vocabulary."""

    CATS = [1, 2, 3, 4]
    ALIASES = {1: {"name": "a", "prompts": ["a", "a-alias"]}, 2: {"name": "b", "prompts": ["b"]},
               3: {"name": "c", "prompts": ["c"]}, 4: {"name": "d", "prompts": ["d"]}}

    def setUp(self):
        self.table = pdet.build_query_table(self.ALIASES)  # queries: a, a-alias, b, c, d
        self.scores = np.array([[0.10, 0.60, 0.30, 0.20, 0.05],
                                [0.02, 0.01, 0.04, 0.50, 0.40],
                                [0.03, 0.02, 0.01, 0.04, 0.05]])
        self.boxes = np.array([[0, 0, 10, 10], [20, 20, 40, 40], [50, 50, 60, 60]], dtype=float)

    def test_raw_scores_ignore_aliases_and_alias_scores_take_the_max(self):
        raw = pdet.category_scores(self.scores, self.table, self.CATS, use_aliases=False)
        alias = pdet.category_scores(self.scores, self.table, self.CATS, use_aliases=True)
        np.testing.assert_allclose(raw[0], [0.10, 0.30, 0.20, 0.05])
        np.testing.assert_allclose(alias[0], [0.60, 0.30, 0.20, 0.05])

    def test_single_block_keeps_one_label_per_box_with_strict_threshold(self):
        cat = pdet.category_scores(self.scores, self.table, self.CATS, use_aliases=False)
        dets = pdet.label_boxes(cat, self.boxes, self.CATS, [self.CATS], min_score=0.05, width=100, height=100)
        # box 0 -> b (0.30); box 1 -> c (0.50); box 2 best is d at 0.05, not > 0.05 (HF uses a strict '>').
        self.assertEqual(sorted((d.category_id, d.score) for d in dets), [(2, 0.30), (3, 0.50)])

    def test_chunked_blocks_allow_one_label_per_block(self):
        cat = pdet.category_scores(self.scores, self.table, self.CATS, use_aliases=True)
        dets = pdet.label_boxes(cat, self.boxes, self.CATS, [[1, 2], [3, 4]], min_score=0.05, width=100, height=100)
        by_box = sorted((d.bbox[0], d.category_id, d.score) for d in dets)
        # box 0 gets one label per block (a via its alias, c); box 1 only c (b = 0.04 is below the floor).
        self.assertEqual(by_box, [(0.0, 1, 0.60), (0.0, 3, 0.20), (20.0, 3, 0.50)])

    def test_label_boxes_clips_to_image_and_tags_source(self):
        cat = np.array([[0.9]])
        dets = pdet.label_boxes(cat, np.array([[-5.0, 2.0, 130.0, 50.0]]), [7], [[7]], min_score=0.05,
                                width=100, height=80, source="owlv2:global")
        self.assertEqual(dets, [Detection(7, 0.9, (0.0, 2.0, 100.0, 50.0), "owlv2:global")])

    def test_variants_cover_the_two_by_two_factorial(self):
        blocks = {"x": [1, 2], "y": [3, 4]}
        out = pdet.extract_variants(self.scores, self.boxes, self.table, self.CATS, blocks, min_score=0.05,
                                    width=100, height=100)
        self.assertEqual(set(out), set(pdet.VARIANTS))
        self.assertEqual(sorted(d.category_id for d in out["single_raw"]), [2, 3])
        self.assertEqual(sorted(d.category_id for d in out["single_alias"]), [1, 3])
        self.assertGreater(len(out["chunked_alias"]), len(out["single_alias"]))


class TestPerClassNmsScales(unittest.TestCase):
    def test_vectorised_nms_matches_greedy_reference(self):
        rng = np.random.default_rng(0)
        dets = []
        for _ in range(300):
            x, y = rng.uniform(0, 80, size=2)
            w, h = rng.uniform(2, 20, size=2)
            dets.append(Detection(int(rng.integers(1, 4)), float(rng.uniform(0.05, 1)), (x, y, x + w, y + h)))
        reference = []
        for det in sorted(dets, key=lambda d: (-d.score, d.category_id, d.bbox)):
            if all(k.category_id != det.category_id or pdet.box_iou(list(k.bbox), list(det.bbox)) < 0.5
                   for k in reference):
                reference.append(det)
        self.assertEqual(pdet.per_class_nms(dets, 0.5), reference)


class TestSiblingSuppression(unittest.TestCase):
    def setUp(self):
        self.groups = pdet.sibling_groups(cfg.CATEGORIES)

    def test_overlapping_siblings_keep_the_higher_score(self):
        dets = [Detection(CAT_ID["watch"], 0.4, (10, 10, 20, 20)), Detection(CAT_ID["clock"], 0.3, (10, 10, 20, 21))]
        kept = pdet.suppress_siblings(dets, self.groups, iou_threshold=0.7)
        self.assertEqual([d.category_id for d in kept], [CAT_ID["watch"]])

    def test_non_siblings_and_separated_siblings_survive(self):
        dets = [Detection(CAT_ID["shoe"], 0.5, (0, 0, 10, 10)), Detection(CAT_ID["boot"], 0.4, (30, 0, 40, 10)),
                Detection(CAT_ID["person"], 0.3, (0, 0, 10, 10)), Detection(CAT_ID["shoe"], 0.2, (0, 0, 10, 10))]
        kept = pdet.suppress_siblings(dets, self.groups, iou_threshold=0.7)
        self.assertEqual(len(kept), 4)  # same-class duplicates are per-class NMS's job, not this rule's


class TestTileGeometry(unittest.TestCase):
    def test_grid_is_four_overlapping_windows_covering_the_image(self):
        tiles = pdet.tile_grid(640, 480, overlap=0.25)
        self.assertEqual(tiles, [(0, 0, 366, 275), (274, 0, 640, 275), (0, 205, 366, 480), (274, 205, 640, 480)])
        tw, th = tiles[0][2], tiles[0][3]
        self.assertAlmostEqual((2 * tw - 640) / tw, 0.25, delta=0.01)  # horizontal overlap = 25% of tile side
        self.assertAlmostEqual((2 * th - 480) / th, 0.25, delta=0.01)
        covered = np.zeros((480, 640), dtype=bool)
        for x1, y1, x2, y2 in tiles:
            covered[y1:y2, x1:x2] = True
        self.assertTrue(covered.all())

    def test_project_box_moves_tile_coordinates_to_the_global_frame(self):
        self.assertEqual(pdet.project_box((10.0, 5.0, 30.0, 25.0), (274, 205, 640, 480)), (284.0, 210.0, 304.0, 230.0))

    def test_inner_edge_cut_is_flagged_but_image_borders_are_not(self):
        tl = (0, 0, 366, 275)
        self.assertTrue(pdet.touches_inner_edge((300.0, 40.0, 366.0, 80.0), tl, 640, 480, margin=3.0))
        self.assertTrue(pdet.touches_inner_edge((300.0, 40.0, 364.0, 80.0), tl, 640, 480, margin=3.0))
        self.assertTrue(pdet.touches_inner_edge((30.0, 200.0, 60.0, 275.0), tl, 640, 480, margin=3.0))
        self.assertFalse(pdet.touches_inner_edge((0.0, 0.0, 50.0, 50.0), tl, 640, 480, margin=3.0))
        self.assertFalse(pdet.touches_inner_edge((100.0, 100.0, 150.0, 150.0), tl, 640, 480, margin=3.0))
        br = (274, 205, 640, 480)
        self.assertFalse(pdet.touches_inner_edge((600.0, 440.0, 640.0, 480.0), br, 640, 480, margin=3.0))
        self.assertTrue(pdet.touches_inner_edge((274.0, 300.0, 300.0, 320.0), br, 640, 480, margin=3.0))

    def test_objects_narrower_than_the_overlap_band_always_have_a_clean_tile(self):
        width, height, margin = 640, 480, 3.0
        tiles = pdet.tile_grid(width, height)
        # the property holds per axis: the horizontal band (92 px) and the vertical band (70 px) differ
        w = (tiles[0][2] - tiles[1][0]) - 2 * margin - 1
        h = (tiles[0][3] - tiles[2][1]) - 2 * margin - 1
        for x in np.arange(0, width - w, 7.0):
            for y in np.arange(0, height - h, 7.0):
                box = (x, y, x + w, y + h)
                clean = [t for t in tiles if t[0] <= x and t[1] <= y and box[2] <= t[2] and box[3] <= t[3]
                         and not pdet.touches_inner_edge(box, t, width, height, margin)]
                self.assertTrue(clean, f"{box} is cut by every tile")


class _FakeVariantDetector:
    """Returns, for every view, one 10x10 box at the view's top-left corner and one cut by the view's right edge."""

    def __call__(self, image):
        w, _ = image.size
        dets = [Detection(1, 0.6, (0.0, 0.0, 10.0, 10.0)), Detection(2, 0.5, (w - 20.0, 30.0, float(w), 50.0))]
        return {"single_raw": dets, "chunked_alias": dets[:1]}


class TestTiledInference(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.image = Image.new("RGB", (640, 480))

    def test_tile_detections_are_projected_and_tagged_by_view(self):
        out = pdet.detect_with_tiles(self.image, _FakeVariantDetector())
        self.assertEqual(set(out), {"single_raw", "chunked_alias"})
        sources = sorted({d.source for d in out["single_raw"]})
        self.assertEqual(sources, ["owlv2:global"] + [f"owlv2:tile{i}" for i in range(4)])
        tile3 = [d for d in out["chunked_alias"] if d.source == "owlv2:tile3"]
        self.assertEqual(tile3, [Detection(1, 0.6, (274.0, 205.0, 284.0, 215.0), "owlv2:tile3")])
        self.assertEqual(pdet.view_index(tile3[0].source), 3)
        self.assertIsNone(pdet.view_index("owlv2:global"))

    def test_merge_keeps_subset_drops_truncated_and_deduplicates(self):
        dets = pdet.detect_with_tiles(self.image, _FakeVariantDetector())["single_raw"]
        merged = pdet.merge_views(dets, 640, 480, tile_category_ids={1, 2})
        tile_cuts = [d for d in merged if d.category_id == 2 and d.source != "owlv2:global"]
        # category 2 is cut at the inner right edge of tiles 0/2 -> dropped; on tiles 1/3 the right edge is the
        # image border -> kept, and it duplicates the global box there (NMS keeps one).
        self.assertTrue(all(pdet.view_index(d.source) in (1, 3) for d in tile_cuts))
        self.assertEqual(len([d for d in merged if d.category_id == 2 and d.bbox == (620.0, 30.0, 640.0, 50.0)]), 1)
        kept_raw = pdet.merge_views(dets, 640, 480, tile_category_ids={1, 2}, drop_truncated=False)
        self.assertGreater(len(kept_raw), len(merged))

    def test_tile_views_have_their_own_threshold_and_size_gate(self):
        dets = [Detection(1, 0.20, (100.0, 100.0, 110.0, 110.0), "owlv2:global"),
                Detection(2, 0.20, (100.0, 100.0, 110.0, 110.0), "owlv2:tile0"),   # below the tile threshold
                Detection(3, 0.40, (100.0, 100.0, 110.0, 110.0), "owlv2:tile0"),   # small -> kept
                Detection(4, 0.40, (10.0, 10.0, 300.0, 250.0), "owlv2:tile0")]     # large (0.23 of the image)
        merged = pdet.merge_views(dets, 640, 480, tile_category_ids={1, 2, 3, 4}, tile_min_score=0.3,
                                  tile_max_area_ratio=0.035)
        self.assertEqual(sorted(d.category_id for d in merged), [1, 3])
        loose = pdet.merge_views(dets, 640, 480, tile_category_ids={1, 2, 3, 4})
        self.assertEqual(sorted(d.category_id for d in loose), [1, 2, 3, 4])  # defaults keep the old behavior

    def test_filter_tile_views_is_merge_views_without_nms(self):
        dets = pdet.detect_with_tiles(self.image, _FakeVariantDetector())["single_raw"]
        kept = pdet.filter_tile_views(dets, 640, 480, tile_category_ids={1, 2})
        self.assertEqual(pdet.per_class_nms(kept), pdet.merge_views(dets, 640, 480, tile_category_ids={1, 2}))
        self.assertGreater(len(kept), len(pdet.merge_views(dets, 640, 480, tile_category_ids={1, 2})))

    def test_merge_ignores_tiles_for_categories_outside_the_subset(self):
        dets = pdet.detect_with_tiles(self.image, _FakeVariantDetector())["single_raw"]
        merged = pdet.merge_views(dets, 640, 480, tile_category_ids=set())
        self.assertEqual({d.source for d in merged}, {"owlv2:global"})


class _FakeTokenizer:
    """Word-level stand-in for BERT: [CLS] + one token per word or '.', + [SEP], with HF-style char offsets."""

    def __call__(self, text, return_offsets_mapping=False):
        import re
        offsets = [(0, 0)] + [m.span() for m in re.finditer(r"[a-z]+|\.", text)] + [(0, 0)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


class TestGroundingDinoPrompt(unittest.TestCase):
    def test_block_prompt_records_one_char_span_per_phrase(self):
        text, spans = pdet.build_block_prompt([(1, "person"), (2, "short pants"), (2, "shorts")])
        self.assertEqual(text, "person. short pants. shorts.")
        self.assertEqual(spans, [(0, 6, 1), (8, 19, 2), (21, 27, 2)])
        self.assertEqual([text[a:b] for a, b, _ in spans], ["person", "short pants", "shorts"])

    def test_tokens_map_to_category_columns_and_separators_to_minus_one(self):
        text, spans = pdet.build_block_prompt([(5, "person"), (9, "short pants")])
        offsets = _FakeTokenizer()(text)["offset_mapping"]  # [CLS] person . short pants . [SEP]
        cols = pdet.token_category_columns(offsets, spans, [9, 5])
        np.testing.assert_array_equal(cols, [-1, 1, -1, 0, 0, -1, -1])

    def test_category_without_tokens_fails_fast(self):
        _, spans = pdet.build_block_prompt([(1, "person")])
        with self.assertRaises(ValueError):
            pdet.token_category_columns([(0, 0), (0, 6), (0, 0)], spans, [1, 2])

    def test_encode_block_enforces_the_text_token_budget(self):
        pairs = [(1, "person"), (2, "shirt"), (3, "short pants")]
        text, cols = pdet.encode_block_prompt(_FakeTokenizer(), pairs, [1, 2, 3], max_text_len=9)
        self.assertEqual(len(cols), 9)
        with self.assertRaises(ValueError):
            pdet.encode_block_prompt(_FakeTokenizer(), pairs, [1, 2, 3], max_text_len=8)

    def test_category_score_is_the_max_over_its_tokens(self):
        probs = np.array([[0.9, 0.2, 0.5, 0.3, 0.7, 0.8, 0.9]])
        cats = pdet.token_scores_to_categories(probs, np.array([-1, 1, -1, 0, 0, -1, -1]), 2)
        np.testing.assert_allclose(cats, [[0.7, 0.2]])

    def test_decoded_boxes_are_scaled_to_the_image_and_labelled_once_per_box(self):
        cols = np.array([-1, 0, -1, 1, -1])
        logits = np.full((3, 7), -9.0)  # padded text positions beyond len(cols) must be ignored
        logits[0, 1], logits[1, 3], logits[2, 1] = 2.0, 0.0, -3.0   # sigmoid: 0.881, 0.5, 0.047
        logits[:, 5:] = 9.0
        boxes = np.array([[0.25, 0.5, 0.5, 1.0], [0.5, 0.5, 0.2, 0.2], [0.9, 0.9, 0.1, 0.1]])  # cxcywh, normalized
        dets = pdet.decode_block_outputs(logits, boxes, cols, [7, 8], min_score=0.1, width=200, height=100,
                                         source="mmgdino")
        self.assertEqual([(d.category_id, round(d.score, 3), d.bbox, d.source) for d in dets],
                         [(7, 0.881, (0.0, 0.0, 100.0, 100.0), "mmgdino"), (8, 0.5, (80.0, 40.0, 120.0, 60.0), "mmgdino")])


class TestCompressedDetectionCache(unittest.TestCase):
    def test_gz_cache_round_trip_with_rounding_keeps_view_sources(self):
        import tempfile
        from pathlib import Path

        from src import pipeline as pl

        dets = {"tpi_0001": [Detection(3, 0.123456789, (1.234567, 2.0, 30.5, 40.0), "owlv2:tile2")], "test_0001": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = pl.save_detection_cache(dets, Path(tmp) / "d.json.gz", meta={"image_sizes": {"x": [4, 3]}},
                                           ndigits=4)
            self.assertEqual(path.read_bytes()[:2], b"\x1f\x8b")  # gzip magic
            loaded, meta = pl.load_detection_cache(path)
        self.assertEqual(loaded, {"tpi_0001": [Detection(3, 0.1235, (1.2346, 2.0, 30.5, 40.0), "owlv2:tile2")],
                                  "test_0001": []})
        self.assertEqual(meta["image_sizes"], {"x": [4, 3]})


class TestPerImageBoxMatching(unittest.TestCase):
    """Per-image TP/FP/FN and per-GT match flags feed the paired bootstrap and the recall-by-size breakdown."""

    GT = {"a": {"height": 100, "width": 100, "categories": [1, 1, 2], "boxes": [[0, 0, 10, 10], [50, 50, 60, 60],
                                                                                [20, 20, 40, 40]],
                "masks": [None] * 3, "pos": {1, 2}, "neg": {3}, "not_exhaustive": {2}},
          "b": {"height": 100, "width": 100, "categories": [1], "boxes": [[0, 0, 10, 10]], "masks": [None],
                "pos": {1}, "neg": set(), "not_exhaustive": set()}}
    PREDS = {"a": [{"category_id": 1, "confidence": 0.9, "bbox": [0, 0, 10, 10]},
                   {"category_id": 1, "confidence": 0.8, "bbox": [1, 0, 10, 10]},   # duplicate -> FP
                   {"category_id": 2, "confidence": 0.7, "bbox": [70, 70, 80, 80]},  # not exhaustive -> ignored
                   {"category_id": 3, "confidence": 0.6, "bbox": [5, 5, 9, 9]},      # negative -> FP
                   {"category_id": 4, "confidence": 0.5, "bbox": [5, 5, 9, 9]}],     # unverified -> ignored
             "b": []}

    def test_counts_and_flags_follow_the_federated_policy(self):
        from src import lvis_evaluator as ev

        per = ev.box_match_details(self.PREDS, self.GT)
        self.assertEqual({k: (v["tp"], v["fp"], v["fn"]) for k, v in per.items()}, {"a": (1, 2, 2), "b": (0, 0, 1)})
        self.assertEqual(per["a"]["gt_matched"], [True, False, False])
        self.assertEqual(per["b"]["gt_matched"], [False])

    def test_summed_counts_equal_the_official_box_metric(self):
        from src import lvis_evaluator as ev

        per = ev.box_match_details(self.PREDS, self.GT)
        report = ev.evaluate_federated(self.PREDS, self.GT, with_masks=False)["box"]
        self.assertEqual(tuple(sum(v[k] for v in per.values()) for k in ("tp", "fp", "fn")),
                         (report["tp"], report["fp"], report["fn"]))


if __name__ == "__main__":
    unittest.main()
