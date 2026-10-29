"""Behavioural tests for T2 (S2): deterministic QA engine and rule-based question interpreter."""

import json
import unittest

import numpy as np

from src import config_and_seeds as cfg
from src import qa_engine as qa
from src import scene_builder as sb

CATS = {1: "sofa", 13: "clock", 17: "person", 112: "ring", 12: "chair", 43: "table"}


def _obj(obj_id, cid, x, y, side=10, conf=0.5, shape=(90, 90)):
    mask = np.zeros(shape, dtype=bool)
    mask[y:y + side, x:x + side] = True
    return sb.build_object(obj_id, cid, conf, [x, y, x + side, y + side], mask)


def _scene(objects, shape=(90, 90)):
    return sb.build_scene_json("synthetic", shape[1], shape[0], objects)


def _q(qtype, classes=(), text="?"):
    return {"question_id": f"T-{qtype}", "type": qtype, "question": text, "class_en": list(classes)}


class TestPredicateExtraction(unittest.TestCase):
    def test_all_canonical_phrases(self):
        cases = {
            '¿La instancia de "reloj" está por encima de la instancia de "anillo"?': "above",
            '¿La instancia de "toldo" está por debajo de la instancia de "vaca"?': "below",
            '¿La instancia de "botella" está a la izquierda de la instancia de "correa"?': "left_of",
            '¿La instancia de "teclado" está a la derecha de la instancia de "maceta"?': "right_of",
            '¿La instancia de "caja" se superpone con la instancia de "mesa"?': "overlaps",
            '¿La instancia de "caja" está superpuesta a la instancia de "mesa"?': "overlaps",
        }
        for text, expected in cases.items():
            self.assertEqual(qa.extract_relation_predicate(text), expected, text)

    def test_unknown_phrase_returns_none_and_answer_is_unsupported(self):
        self.assertIsNone(qa.extract_relation_predicate('¿La "silla" está cerca de la "mesa"?'))
        scene = _scene([_obj("a", 12, 0, 0), _obj("b", 43, 50, 50)])
        ans = qa.resolve_deterministic_qa(scene, _q("relation", ["chair", "table"], '¿La "silla" está cerca de la "mesa"?'), CATS)
        self.assertEqual((ans["normalized_answer"], ans["evidence"]), ("unsupported", []))


class TestEmptyScene(unittest.TestCase):
    def setUp(self):
        self.scene = _scene([])

    def test_closed_types(self):
        expected = {"count": 0, "existence": "no", "position": "insufficient_information",
                    "size": "insufficient_information"}
        for qtype, value in expected.items():
            ans = qa.resolve_deterministic_qa(self.scene, _q(qtype, ["person"]), CATS)
            self.assertEqual(ans["normalized_answer"], value, qtype)
            self.assertIs(type(ans["normalized_answer"]), type(value), qtype)
            self.assertEqual(ans["evidence"], [], qtype)

    def test_description_reports_absence(self):
        ans = qa.resolve_deterministic_qa(self.scene, _q("description"), CATS)
        self.assertIn("No se detectaron objetos", ans["answer"])
        self.assertEqual(ans["evidence"], [])

    def test_answers_validate_against_schema(self):
        schema = json.loads(cfg.get_dataset_paths().schema.read_text())
        questions = [_q(t, ["person"]) for t in ("count", "existence", "position", "size", "description")]
        scene = dict(self.scene, questions=qa.answer_questions(self.scene, questions, CATS))
        self.assertEqual(sb.validate_scene(scene, schema), [])


class TestClosedQuestions(unittest.TestCase):
    def setUp(self):
        self.scene = _scene([
            _obj("p1", 17, 5, 5, conf=0.4), _obj("p2", 17, 70, 70, side=15, conf=0.9),
            _obj("p3", 17, 40, 40, side=12, conf=0.9), _obj("c1", 13, 5, 70, conf=0.8),
        ])

    def test_count_is_int_with_unique_evidence(self):
        ans = qa.resolve_deterministic_qa(self.scene, _q("count", ["person"]), CATS)
        self.assertIs(type(ans["normalized_answer"]), int)
        self.assertEqual(ans["normalized_answer"], 3)
        self.assertEqual(sorted(ans["evidence"]), ["p1", "p2", "p3"])

    def test_existence(self):
        self.assertEqual(qa.resolve_deterministic_qa(self.scene, _q("existence", ["clock"]), CATS)["normalized_answer"], "yes")
        ans = qa.resolve_deterministic_qa(self.scene, _q("existence", ["sofa"]), CATS)
        self.assertEqual((ans["normalized_answer"], ans["evidence"]), ("no", []))

    def test_position_and_size_pick_max_confidence_then_max_area(self):
        pos = qa.resolve_deterministic_qa(self.scene, _q("position", ["person"]), CATS)
        self.assertEqual((pos["normalized_answer"], pos["evidence"]), ("bottom_right", ["p2"]))
        size = qa.resolve_deterministic_qa(self.scene, _q("size", ["person"]), CATS)
        self.assertEqual(size["evidence"], ["p2"])
        self.assertEqual(size["normalized_answer"], cfg.size_from_area_ratio(225 / 8100))

    def test_relation_yes_records_both_ids_and_no_has_empty_evidence(self):
        text_above = '¿La instancia de "persona" está por encima de la instancia de "reloj"?'
        yes = qa.resolve_deterministic_qa(self.scene, _q("relation", ["person", "clock"], text_above), CATS)
        self.assertEqual(yes["normalized_answer"], "yes")
        self.assertEqual(len(yes["evidence"]), 2)
        self.assertIn("c1", yes["evidence"])
        text_left = '¿La instancia de "reloj" está a la derecha de la instancia de "persona"?'
        no = qa.resolve_deterministic_qa(self.scene, _q("relation", ["clock", "person"], text_left), CATS)
        self.assertEqual((no["normalized_answer"], no["evidence"]), ("no", []))

    def test_relation_with_missing_class_is_no(self):
        text = '¿La instancia de "sofá" está a la izquierda de la instancia de "reloj"?'
        ans = qa.resolve_deterministic_qa(self.scene, _q("relation", ["sofa", "clock"], text), CATS)
        self.assertEqual((ans["normalized_answer"], ans["evidence"]), ("no", []))
        self.assertIn("sofá", ans["answer"])

    def test_unknown_category_and_type_are_unsupported(self):
        for q in (_q("count", ["unicorn"]), _q("color", ["person"])):
            ans = qa.resolve_deterministic_qa(self.scene, q, CATS)
            self.assertEqual((ans["normalized_answer"], ans["evidence"]), ("unsupported", []))

    def test_description_is_spanish_with_evidence_subset(self):
        ans = qa.resolve_deterministic_qa(self.scene, _q("description"), CATS)
        self.assertIn("persona", ans["answer"])
        self.assertTrue(set(ans["evidence"]) <= {"p1", "p2", "p3", "c1"})
        self.assertEqual(len(ans["evidence"]), len(set(ans["evidence"])))
        self.assertEqual(ans, qa.resolve_deterministic_qa(self.scene, _q("description"), CATS))


class TestRuleBasedInterpreter(unittest.TestCase):
    def test_lexicon_covers_all_categories_without_ambiguity(self):
        self.assertEqual(sorted(cfg.CATEGORIES_ES), list(range(1, 121)))
        with self.assertRaises(ValueError):
            cfg.build_spanish_names({1: "sofa", 2: "laptop"}, {"sofa": ["x"], "laptop": ["x"]})

    def test_interprets_text_only(self):
        iq = qa.interpret_question_text('¿Cuántas instancias de la categoría "guante de béisbol" aparecen en la imagen?',
                                        cfg.CATEGORIES)
        self.assertEqual((iq["type"], iq["class_en"], iq["status"]), ("count", ["baseball glove"], "ok"))
        iq = qa.interpret_question_text('¿La instancia de "reloj" está por encima de la instancia de "anillo"?',
                                        cfg.CATEGORIES)
        self.assertEqual((iq["type"], iq["class_en"], iq["predicate"]), ("relation", ["clock", "ring"], "above"))

    def test_out_of_vocabulary_or_out_of_scope_is_unsupported(self):
        for text in ('¿Cuántos "unicornios" hay en la imagen?', "¿De qué color es el cielo?"):
            iq = qa.interpret_question_text(text, cfg.CATEGORIES)
            self.assertEqual(iq["status"], "unsupported", text)
            ans = qa.resolve_llm_mode_qa(_scene([]), "X-1", text, cfg.CATEGORIES)
            self.assertEqual(ans["normalized_answer"], "unsupported")
            self.assertIn("interpreted_query", ans)


def _load_questions():
    path = cfg.get_dataset_paths().questions_dev
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# GT-derived expectations for the 40 closed questions (from annotations.json geometry, see handoff T2 notes).
EXPECTED_GT = {
    "DEV-1-Q1": 2, "DEV-2-Q1": 2, "DEV-3-Q1": 2, "DEV-4-Q1": 4, "DEV-6-Q1": 2, "DEV-5-Q1": 3,
    "DEV-7-Q1": 2, "DEV-8-Q1": 2, "DEV-9-Q1": 2, "DEV-10-Q1": 2,
    "DEV-1-Q2": "yes", "DEV-2-Q2": "yes", "DEV-3-Q2": "no", "DEV-4-Q2": "no", "DEV-6-Q2": "no",
    "DEV-11-Q1": "yes", "DEV-12-Q1": "yes", "DEV-14-Q1": "yes", "DEV-20-Q1": "no", "DEV-23-Q1": "no",
    "DEV-11-Q2": "bottom_left", "DEV-12-Q2": "center", "DEV-14-Q2": "middle_left", "DEV-20-Q2": "middle_right",
    "DEV-23-Q2": "center",
    "DEV-15-Q2": "small", "DEV-19-Q2": "small", "DEV-22-Q2": "small", "DEV-24-Q2": "large", "DEV-25-Q2": "medium",
    "DEV-5-Q2": "yes", "DEV-7-Q2": "yes", "DEV-8-Q2": "yes", "DEV-9-Q2": "no", "DEV-10-Q2": "no",
    "DEV-15-Q1": "yes", "DEV-19-Q1": "yes", "DEV-22-Q1": "no", "DEV-24-Q1": "no", "DEV-25-Q1": "no",
}


class TestGroundTruthOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.build_gt_oracle import build_gt_scenes

        paths = cfg.get_dataset_paths()
        annotations = json.loads(paths.annotations.read_text(encoding="utf-8"))
        cls.questions = _load_questions()
        cls.scenes = build_gt_scenes(paths, annotations, cfg.load_qa_image_ids(paths.questions_dev))
        cls.schema = json.loads(paths.schema.read_text(encoding="utf-8"))

    def _key(self, q):
        return q["image_file"].rsplit(".", 1)[0]

    def test_60_deterministic_answers_match_gt_expectations(self):
        self.assertEqual(len(self.questions), 60)
        for q in self.questions:
            scene = self.scenes[self._key(q)]
            ans = qa.resolve_deterministic_qa(scene, q, cfg.CATEGORIES)
            ids = {o["id"] for o in scene["objects"]}
            self.assertTrue(set(ans["evidence"]) <= ids, q["question_id"])
            if q["type"] == "description":
                self.assertTrue(ans["answer"] and ans["evidence"], q["question_id"])
                continue
            expected = EXPECTED_GT[q["question_id"]]
            self.assertEqual(ans["normalized_answer"], expected, q["question_id"])
            self.assertIs(type(ans["normalized_answer"]), type(expected), q["question_id"])

    def test_rule_based_llm_mode_agrees_on_40_closed_questions(self):
        for q in self.questions:
            scene = self.scenes[self._key(q)]
            ans = qa.resolve_llm_mode_qa(scene, q["question_id"], q["question"], cfg.CATEGORIES)
            self.assertEqual(ans["interpreted_query"]["type"], q["type"], q["question_id"])
            if q["type"] != "description":
                self.assertEqual(ans["normalized_answer"], EXPECTED_GT[q["question_id"]], q["question_id"])

    def test_scenes_with_both_question_fields_validate(self):
        by_image = {}
        for q in self.questions:
            by_image.setdefault(self._key(q), []).append(q)
        for key, qs in by_image.items():
            scene = dict(self.scenes[key])
            scene["questions"] = qa.answer_questions(scene, qs, cfg.CATEGORIES)
            scene["questions_llm"] = qa.answer_questions_llm(scene, qs, cfg.CATEGORIES)
            self.assertEqual(len(scene["questions"]), 3)
            self.assertEqual(sb.validate_scene(scene, self.schema), [], key)


if __name__ == "__main__":
    unittest.main()
