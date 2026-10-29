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


if __name__ == "__main__":
    unittest.main()
