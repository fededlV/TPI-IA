"""Deterministic QA engine over SceneJSON and a rule-based question-text interpreter (questions_llm fallback).

Epistemic contract: answers use only SceneJSON (observed: boxes, scores, masks; derived: geometry, relations).
Anything the scene cannot support is answered "unsupported" or "insufficient_information".
"""

from __future__ import annotations

import re
import unicodedata

from src import config_and_seeds as cfg
from src import scene_builder as sb

UNSUPPORTED = "unsupported"
INSUFFICIENT = "insufficient_information"
SUPPORTED_TYPES = ("description", "count", "existence", "position", "size", "relation")
DIRECTIONAL = ("left_of", "right_of", "above", "below")

# Canonical Spanish phrasings -> relation predicate (matched on accent-stripped lowercase text).
PREDICATE_PHRASES = (
    ("por encima de", "above"), ("encima de", "above"),
    ("por debajo de", "below"), ("debajo de", "below"),
    ("a la izquierda de", "left_of"), ("a la derecha de", "right_of"),
    ("se superpone", "overlaps"), ("superpuesta", "overlaps"), ("superpuesto", "overlaps"),
)
POSITION_ES = {
    "top_left": "superior izquierda", "top_center": "superior central", "top_right": "superior derecha",
    "middle_left": "media izquierda", "center": "central", "middle_right": "media derecha",
    "bottom_left": "inferior izquierda", "bottom_center": "inferior central", "bottom_right": "inferior derecha",
}
SIZE_ES = {"small": "pequeño", "medium": "mediano", "large": "grande"}
PREDICATE_ES = {"left_of": "a la izquierda de", "right_of": "a la derecha de", "above": "por encima de",
                "below": "por debajo de", "overlaps": "superpuesta con"}
ABSENCE_CONFIDENCE = 0.5  # absence of detections is weak evidence (federated, non-exhaustive detection)
MAX_DESCRIBED_CATEGORIES = 5
MAX_DESCRIBED_PAIRS = 3


def normalize_text(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in stripped if not unicodedata.combining(ch)).strip()


def extract_relation_predicate(question_text: str) -> str | None:
    text = normalize_text(question_text)
    for phrase, predicate in PREDICATE_PHRASES:
        if phrase in text:
            return predicate
    return None


def spanish_name(category_id: int, categories_map: dict[int, str]) -> str:
    names = cfg.CATEGORIES_ES.get(category_id)
    return names[0] if names else categories_map.get(category_id, str(category_id))


# --------------------------------------------------------------------------- scene helpers


def _instances(scene: dict, category_id: int) -> list[dict]:
    """Instances of a category, ranked by confidence, then area, then id (deterministic)."""
    objs = [o for o in scene.get("objects", []) if o["category_id"] == category_id]
    return sorted(objs, key=lambda o: (-o["confidence"], -o["area_ratio"], o["id"]))


def _mean_confidence(objs: list[dict]) -> float:
    return round(sum(o["confidence"] for o in objs) / len(objs), 4) if objs else ABSENCE_CONFIDENCE


def _relation_holds(scene: dict, triples: set, a: dict, b: dict, predicate: str) -> bool:
    """Look the triple up in scene relations; fall back to geometry for pairs beyond the relation cap."""
    if (a["id"], predicate, b["id"]) in triples:
        return True
    (ax, ay), (bx, by) = a["centroid"], b["centroid"]
    if predicate == "left_of":
        return ax < bx
    if predicate == "right_of":
        return ax > bx
    if predicate == "above":
        return ay < by
    if predicate == "below":
        return ay > by
    if sb._boxes_intersect(a["bbox"], b["bbox"]):
        return True
    ma, mb = sb.decode_rle(a["mask"]), sb.decode_rle(b["mask"])
    return sb._masks_intersect(ma, mb, sb.mask_tight_bbox(ma), sb.mask_tight_bbox(mb))


def _result(q: dict, answer: str, normalized, evidence: list[str], confidence: float, reason: str) -> dict:
    # The schema enum needs a type; out-of-scope queries (unknown type) are labelled as free text.
    qtype = q.get("type") if q.get("type") in SUPPORTED_TYPES else "description"
    return {"question_id": str(q.get("question_id", "")), "type": qtype, "question": q.get("question") or "?",
            "answer": answer,
            "normalized_answer": normalized, "evidence": list(dict.fromkeys(evidence)),
            "confidence": float(min(max(confidence, 0.0), 1.0)), "reason": reason}


def _unsupported(q: dict, reason: str) -> dict:
    return _result(q, "La consulta no puede responderse con el vocabulario o la información del SceneJSON.",
                   UNSUPPORTED, [], 0.0, reason)


# --------------------------------------------------------------------------- answer types


def _answer_count(scene, q, cid, name_es):
    objs = _instances(scene, cid)
    n = len(objs)
    return _result(q, f'Se detectaron {n} instancia(s) de "{name_es}" en la escena.', int(n),
                   [o["id"] for o in objs], _mean_confidence(objs), "count_of_detected_instances")


def _answer_existence(scene, q, cid, name_es):
    objs = _instances(scene, cid)
    if not objs:
        return _result(q, f'No se detectó ninguna instancia de "{name_es}".', "no", [], ABSENCE_CONFIDENCE,
                       "no_detected_instance")
    return _result(q, f'Sí, se detectaron {len(objs)} instancia(s) de "{name_es}".', "yes",
                   [o["id"] for o in objs], _mean_confidence(objs), "detected_instances")


def _answer_attribute(scene, q, cid, name_es, attribute):
    objs = _instances(scene, cid)
    if not objs:
        return _result(q, f'No se detectó ninguna instancia de "{name_es}"; no hay información suficiente.',
                       INSUFFICIENT, [], 0.0, "no_detected_instance")
    top = objs[0]
    value = top[attribute]
    if attribute == "position":
        text = f'La instancia de "{name_es}" de mayor confianza está en la zona {POSITION_ES[value]} de la imagen.'
    else:
        text = (f'La instancia de "{name_es}" de mayor confianza tiene tamaño relativo {SIZE_ES[value]} '
                f'(area_ratio={top["area_ratio"]:.5f}).')
    return _result(q, text, value, [top["id"]], top["confidence"], f"highest_confidence_of_{len(objs)}")


def _answer_relation(scene, q, cids, names_es, predicate):
    subjects, objects = _instances(scene, cids[0]), _instances(scene, cids[1])
    missing = [names_es[i] for i, objs in enumerate((subjects, objects)) if not objs]
    if missing:
        quoted = ", ".join(f'"{m}"' for m in missing)
        return _result(q, f"No: no se detectó ninguna instancia de {quoted}.", "no", [], ABSENCE_CONFIDENCE,
                       "missing_class")
    triples = {(r["subject"], r["predicate"], r["object"]) for r in scene.get("relations", [])}
    pairs = sorted(((a, b) for a in subjects for b in objects if a["id"] != b["id"]),
                   key=lambda p: (-(p[0]["confidence"] + p[1]["confidence"]), p[0]["id"], p[1]["id"]))
    for a, b in pairs:
        if _relation_holds(scene, triples, a, b, predicate):
            return _result(q, f'Sí: "{names_es[0]}" ({a["id"]}) está {PREDICATE_ES[predicate]} '
                              f'"{names_es[1]}" ({b["id"]}).', "yes", [a["id"], b["id"]],
                           _mean_confidence([a, b]), "relation_found")
    conf = _mean_confidence(list(pairs[0])) if pairs else ABSENCE_CONFIDENCE
    return _result(q, f'No: ningún par de "{names_es[0]}" y "{names_es[1]}" cumple la relación '
                      f'"{PREDICATE_ES[predicate]}".', "no", [], conf, "relation_not_found")


def describe_scene(scene: dict, categories_map: dict[int, str]) -> tuple[str, list[str]]:
    """Deterministic Spanish description from objects and relations; returns (text, mentioned ids)."""
    objects = scene.get("objects", [])
    if not objects:
        return ("No se detectaron objetos en la escena; el SceneJSON no aporta información visual para "
                "describirla."), []
    by_cat: dict[int, list[dict]] = {}
    for o in objects:
        by_cat.setdefault(o["category_id"], []).append(o)
    ranked_cats = sorted(by_cat, key=lambda c: (-len(by_cat[c]), -max(o["confidence"] for o in by_cat[c]), c))
    top_cats = ranked_cats[:MAX_DESCRIBED_CATEGORIES]
    listing = ", ".join(f"{spanish_name(c, categories_map)} ({len(by_cat[c])})" for c in top_cats)
    sentences = [f"Se detectaron {len(objects)} objeto(s) de {len(by_cat)} categoría(s).",
                 f"Categorías más frecuentes: {listing}."]
    evidence = [o["id"] for c in top_cats for o in sorted(by_cat[c], key=lambda o: -o["confidence"])]

    leaders = [_instances(scene, c)[0] for c in top_cats]
    lead = leaders[0]
    sentences.append(f'La instancia de mayor confianza de "{spanish_name(lead["category_id"], categories_map)}" '
                     f'(la categoría más frecuente) se ubica en la zona {POSITION_ES[lead["position"]]} y es de '
                     f'tamaño {SIZE_ES[lead["size"]]}.')
    triples = {(r["subject"], r["predicate"], r["object"]) for r in scene.get("relations", [])}
    facts = []
    for i in range(len(leaders)):
        for j in range(i + 1, len(leaders)):
            a, b = leaders[i], leaders[j]
            preds = [p for p in DIRECTIONAL if (a["id"], p, b["id"]) in triples]
            if not preds:
                continue
            phrase = " y ".join(PREDICATE_ES[p] for p in preds[:2]).replace(" de y ", " y ")
            facts.append(f'"{spanish_name(a["category_id"], categories_map)}" está {phrase} '
                         f'"{spanish_name(b["category_id"], categories_map)}"')
            if len(facts) == MAX_DESCRIBED_PAIRS:
                break
        if len(facts) == MAX_DESCRIBED_PAIRS:
            break
    if facts:
        sentences.append("Relaciones entre las instancias de mayor confianza de cada categoría: "
                         + "; ".join(facts) + ".")
    return " ".join(sentences), list(dict.fromkeys(evidence))


# --------------------------------------------------------------------------- public API


def resolve_deterministic_qa(scene_json: dict, question_entry: dict, categories_map: dict[int, str]) -> dict:
    """Answer one questions_dev.jsonl entry using only the SceneJSON and the structured question fields."""
    q = question_entry
    qtype = q.get("type")
    if qtype not in SUPPORTED_TYPES:
        return _unsupported(q, f"unsupported_type:{qtype}")
    if qtype == "description":
        text, evidence = describe_scene(scene_json, categories_map)
        conf = _mean_confidence([o for o in scene_json.get("objects", []) if o["id"] in set(evidence)])
        return _result(q, text, text, evidence, conf if evidence else 1.0, "template_description")

    name_to_id = {name: cid for cid, name in categories_map.items()}
    classes = list(q.get("class_en") or [])
    cids = [name_to_id.get(c) for c in classes]
    expected = 2 if qtype == "relation" else 1
    if len(cids) != expected or None in cids:
        return _unsupported(q, f"unknown_or_missing_classes:{classes}")
    names_es = [spanish_name(c, categories_map) for c in cids]
    if qtype == "count":
        return _answer_count(scene_json, q, cids[0], names_es[0])
    if qtype == "existence":
        return _answer_existence(scene_json, q, cids[0], names_es[0])
    if qtype in ("position", "size"):
        return _answer_attribute(scene_json, q, cids[0], names_es[0], qtype)
    predicate = extract_relation_predicate(q.get("question", ""))
    if predicate is None:
        return _unsupported(q, "unknown_relation_predicate")
    return _answer_relation(scene_json, q, cids, names_es, predicate)


def answer_questions(scene: dict, questions: list[dict], categories_map: dict[int, str] | None = None) -> list[dict]:
    """Deterministic mode for all questions of one image (also the suite's qa_fn hook)."""
    categories_map = categories_map or cfg.CATEGORIES
    return [resolve_deterministic_qa(scene, q, categories_map) for q in questions]


# --------------------------------------------------------------------------- rule-based interpreter (LLM-mode fallback)

_TYPE_PATTERNS = (
    ("description", r"\bdescrib"), ("count", r"\bcuant[oa]s\b"), ("size", r"que tamano"),
    ("position", r"(en que zona|donde esta|en que parte)"), ("existence", r"(hay al menos|existe|hay alguna?)"),
)
_QUOTED = re.compile(r"[\"«“](.+?)[\"»”]")


def _build_term_index(categories_map: dict[int, str]) -> dict[str, int]:
    index = {normalize_text(name): cid for cid, name in categories_map.items()}
    for cid, names in cfg.CATEGORIES_ES.items():
        if cid in categories_map:
            index.update({normalize_text(n): cid for n in names})
    return index


def interpret_question_text(question_text: str, categories_map: dict[int, str]) -> dict:
    """Image-blind interpretation from raw text + vocabulary only (never structured question fields)."""
    text = normalize_text(question_text)
    query = {"source": "rule_based_fallback", "status": "ok", "type": None, "class_en": [], "category_ids": [],
             "predicate": None, "reason": None}
    predicate = extract_relation_predicate(question_text)
    qtype = next((t for t, pattern in _TYPE_PATTERNS if re.search(pattern, text)), None)
    if predicate is not None and qtype is None:
        qtype = "relation"
    query.update(type=qtype, predicate=predicate if qtype == "relation" else None)
    if qtype is None:
        return dict(query, status=UNSUPPORTED, reason="out_of_scope_question")
    if qtype == "description":
        return query
    index = _build_term_index(categories_map)
    terms = [normalize_text(t) for t in _QUOTED.findall(question_text)]
    cids = [index.get(t) for t in terms]
    expected = 2 if qtype == "relation" else 1
    if len(cids) != expected or None in cids:
        return dict(query, status=UNSUPPORTED, reason=f"out_of_vocabulary_or_ambiguous:{terms}")
    return dict(query, category_ids=cids, class_en=[categories_map[c] for c in cids])


def resolve_llm_mode_qa(scene: dict, question_id: str, question_text: str, categories_map: dict[int, str]) -> dict:
    """LLM-mode answer: interpret text only, then delegate to the deterministic engine (never the reverse)."""
    interpreted = interpret_question_text(question_text, categories_map)
    entry = {"question_id": question_id, "question": question_text, "type": interpreted["type"],
             "class_en": interpreted["class_en"]}
    if interpreted["status"] != "ok":
        answer = _unsupported(entry, interpreted["reason"])
    else:
        answer = resolve_deterministic_qa(scene, entry, categories_map)
    answer["interpreted_query"] = interpreted
    return answer


def answer_questions_llm(scene: dict, questions: list[dict], categories_map: dict[int, str] | None = None) -> list[dict]:
    """LLM mode for all questions of one image; only question_id and raw question text are read."""
    categories_map = categories_map or cfg.CATEGORIES
    return [resolve_llm_mode_qa(scene, q["question_id"], q["question"], categories_map) for q in questions]
