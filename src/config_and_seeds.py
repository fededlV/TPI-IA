"""Global configuration: seeds, dataset paths, categories, prompt aliases, size thresholds and CAL/VAL split."""

from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEED = 42

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
SPLIT_PATH = ARTIFACTS_DIR / "split_cal_val.json"

LOCAL_DATASET_DIR = Path("/home/fede/Ingenieria_en_Sistemas/Inteligencia Artificial/Tp/dataset")
KAGGLE_INPUT_DIR = Path("/kaggle/input")
DATASET_MARKER = "scene_schema.json"

# Canonical size thresholds over mask area_ratio (COCO 32^2 / 96^2 relative to ~640x480).
SMALL_MAX_AREA_RATIO = 0.0035
MEDIUM_MAX_AREA_RATIO = 0.0350
SIZE_THRESHOLDS = (SMALL_MAX_AREA_RATIO, MEDIUM_MAX_AREA_RATIO)

N_CAL = 80
N_VAL = 40
N_QA_PER_SPLIT = 10


def set_global_seeds(seed: int = SEED) -> None:
    """Fix seeds for random, numpy and torch (when installed)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device() -> str:
    """Inference device: TPI_DEVICE override, else CUDA when available (Colab/Kaggle), else CPU."""
    override = os.environ.get("TPI_DEVICE")
    if override:
        return override
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- paths


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @property
    def schema(self) -> Path:
        return self.root / "scene_schema.json"

    @property
    def sample_submission(self) -> Path:
        return self.root / "sample_submission.csv"

    @property
    def competition_dir(self) -> Path:
        return self.root / "tpi_ia_2026_competition_public"

    @property
    def categories(self) -> Path:
        return self.competition_dir / "categories.csv"

    @property
    def test_images(self) -> Path:
        return self.competition_dir / "test_images.csv"

    @property
    def test_images_dir(self) -> Path:
        return self.competition_dir / "test_images"

    @property
    def questions_dev(self) -> Path:
        return self.root / "tpi_2026_qa_public" / "questions_dev.jsonl"

    @property
    def annotations(self) -> Path:
        return self.root / "development" / "annotations.json"

    @property
    def dev_images_dir(self) -> Path:
        return self.root / "development" / "images"


def _candidate_roots() -> list[Path]:
    candidates = []
    env_dir = os.environ.get("TPI_DATASET_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    if KAGGLE_INPUT_DIR.is_dir():
        candidates.extend(sorted(p.parent for p in KAGGLE_INPUT_DIR.glob(f"**/{DATASET_MARKER}")))
    candidates.append(LOCAL_DATASET_DIR)
    candidates.append(REPO_ROOT / "Dataset")
    return candidates


def resolve_dataset_root() -> Path:
    """Return the first candidate directory containing the dataset marker file."""
    candidates = _candidate_roots()
    for root in candidates:
        if (root / DATASET_MARKER).is_file():
            return root
    raise FileNotFoundError(f"Dataset not found; looked for {DATASET_MARKER} in: {[str(c) for c in candidates]}")


def get_dataset_paths() -> DatasetPaths:
    return DatasetPaths(resolve_dataset_root())


# --------------------------------------------------------------------------- categories & aliases


def load_categories(path: Path) -> dict[int, str]:
    with open(path, newline="", encoding="utf-8") as f:
        categories = {int(row["category_id"]): row["category_name"].strip() for row in csv.DictReader(f)}
    if sorted(categories) != list(range(1, 121)):
        raise ValueError(f"Expected category ids 1..120, got {len(categories)} ids")
    return categories


# Visual-friendly synonyms for archaic or ambiguous LVIS names, keyed by official name. Every prompt must name
# exactly one category (no "door handle" under both handle and doorknob, no bare "glasses"/"mouse"/"cap"/"sign").
_ALIAS_OVERRIDES: dict[str, list[str]] = {
    "fireplug": ["fire hydrant"],
    "spectacles": ["eyeglasses"],
    "television set": ["tv", "television"],
    "trash can": ["garbage can", "trash bin"],
    "doorknob": ["door knob", "round door knob"],
    "wall socket": ["electrical outlet", "power socket"],
    "flowerpot": ["plant pot", "flower pot"],
    "signboard": ["shop sign", "billboard"],
    "trousers": ["pants"],
    "short pants": ["shorts"],
    "jean": ["jeans", "denim pants"],
    "handle": ["door handle", "drawer handle"],
    "knob": ["cabinet knob", "control knob"],
    "cell phone": ["mobile phone", "smartphone"],
    "computer monitor": ["monitor", "computer screen"],
    "computer keyboard": ["keyboard"],
    "computer mouse": ["wireless mouse"],
    "drinking glass": ["water glass", "tumbler"],
    "flower arrangement": ["bouquet", "flowers in vase"],
    "streetlight": ["street lamp", "lamppost"],
    "taillight": ["tail light", "car rear light"],
    "headlight": ["car headlight"],
    "necktie": ["tie"],
    "jersey": ["sports jersey"],
    "tag": ["label tag", "price tag"],
    "strap": ["shoulder strap", "bag strap"],
    "vent": ["air vent"],
    "awning": ["canopy"],
    "banner": ["hanging banner"],
    "sofa": ["couch"],
    "handbag": ["purse"],
    "watch": ["wristwatch"],
    "baseball cap": ["sports cap"],
    "ski boot": ["ski boots"],
    "plastic bag": ["shopping bag"],
    "teddy bear": ["stuffed toy bear"],
    "windshield wiper": ["wiper blade"],
    "license plate": ["number plate"],
    "street sign": ["road sign"],
    "button": ["clothing button"],
    "pole": ["metal pole"],
    "motorcycle": ["motorbike"],
    "airplane": ["plane", "aircraft"],
    "bicycle": ["bike"],
    "refrigerator": ["fridge"],
}


def build_prompt_aliases(categories: dict[int, str],
                         overrides: dict[str, list[str]] | None = None) -> dict[int, dict]:
    """Map every category_id to its official name and an ordered, de-duplicated prompt list."""
    overrides = _ALIAS_OVERRIDES if overrides is None else overrides
    known = set(categories.values())
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"Alias overrides reference unknown categories: {unknown}")
    aliases, owner = {}, {}
    for cid, name in sorted(categories.items()):
        prompts = list(dict.fromkeys([name] + overrides.get(name, [])))
        for prompt in prompts:
            if owner.setdefault(prompt, cid) != cid:
                raise ValueError(f"Prompt {prompt!r} is ambiguous: categories {owner[prompt]} and {cid}")
        aliases[cid] = {"name": name, "prompts": prompts}
    return aliases


# Generic Spanish vocabulary for the 120 categories (first entry = canonical Spanish name), keyed by official
# name. Used to verbalize answers and to map Spanish question text to category ids without structured fields.
CATEGORY_NAMES_ES: dict[str, list[str]] = {
    "sofa": ["sofá", "sillón"], "laptop": ["computadora portátil", "laptop", "notebook"],
    "pillow": ["almohada", "almohadón", "cojín"], "bench": ["banco", "banca"],
    "jacket": ["chaqueta", "campera", "chamarra"], "signboard": ["cartel", "letrero"], "backpack": ["mochila"],
    "traffic light": ["semáforo"], "umbrella": ["paraguas", "sombrilla"], "horse": ["caballo"],
    "handle": ["mango", "manija", "asa"], "chair": ["silla"], "clock": ["reloj"], "napkin": ["servilleta"],
    "pizza": ["pizza"], "flag": ["bandera"], "person": ["persona"], "trousers": ["pantalones", "pantalón"],
    "trash can": ["cesto de basura", "tacho de basura", "bote de basura", "papelera"],
    "sock": ["calcetín", "media"], "vase": ["jarrón", "florero"], "bicycle": ["bicicleta"],
    "tennis racket": ["raqueta de tenis"], "necklace": ["collar"],
    "cell phone": ["teléfono celular", "celular", "teléfono móvil"], "book": ["libro"],
    "sunglasses": ["gafas de sol", "anteojos de sol", "lentes de sol"], "lamp": ["lámpara"],
    "bottle": ["botella"], "glove": ["guante"], "shoe": ["zapato"], "blanket": ["manta", "frazada", "cobija"],
    "computer keyboard": ["teclado de computadora", "teclado"], "dog": ["perro"],
    "jean": ["jean", "jeans", "vaquero"], "short pants": ["pantalones cortos", "pantalón corto", "shorts"],
    "flower arrangement": ["arreglo floral", "ramo de flores"], "ski": ["esquí"], "cat": ["gato"],
    "wheel": ["rueda"], "faucet": ["grifo", "canilla"], "helmet": ["casco"], "table": ["mesa"],
    "drinking glass": ["vaso"], "headlight": ["faro delantero", "faro"],
    "streetlight": ["farola", "poste de luz", "farol"], "television set": ["televisor", "televisión", "tele"],
    "belt": ["cinturón"], "bird": ["pájaro", "ave"], "knife": ["cuchillo"], "watch": ["reloj de pulsera"],
    "handbag": ["bolso", "cartera"], "street sign": ["señal de tránsito", "señal de calle", "cartel de calle"],
    "bowl": ["cuenco", "tazón", "bol"], "ski pole": ["bastón de esquí"], "hat": ["sombrero"],
    "plate": ["plato"], "car": ["auto", "coche", "carro"], "curtain": ["cortina"], "elephant": ["elefante"],
    "bed": ["cama"], "drawer": ["cajón"], "earring": ["arete", "pendiente", "aro"],
    "boat": ["bote", "barco", "embarcación"], "necktie": ["corbata"], "skateboard": ["patineta", "monopatín"],
    "wall socket": ["tomacorriente", "enchufe de pared", "enchufe"], "fork": ["tenedor"],
    "bus": ["autobús", "colectivo", "ómnibus"], "giraffe": ["jirafa"], "toilet": ["inodoro", "retrete"],
    "surfboard": ["tabla de surf"], "motorcycle": ["motocicleta", "moto"], "cabinet": ["gabinete", "armario"],
    "taillight": ["luz trasera", "faro trasero"], "train": ["tren"], "baseball cap": ["gorra de béisbol", "gorra"],
    "airplane": ["avión"], "jersey": ["camiseta deportiva", "jersey", "camiseta"], "pole": ["poste"],
    "doorknob": ["pomo de puerta", "picaporte", "perilla de puerta"],
    "baseball bat": ["bate de béisbol", "bate"], "shirt": ["camisa"],
    "license plate": ["matrícula", "placa", "patente"], "banana": ["banana", "plátano"], "mirror": ["espejo"],
    "kite": ["cometa", "barrilete", "papalote"], "frisbee": ["frisbee", "disco volador"],
    "sink": ["lavabo", "lavamanos", "pileta", "fregadero"], "spectacles": ["gafas", "anteojos", "lentes"],
    "button": ["botón"], "goggles": ["gafas protectoras", "antiparras", "goggles"], "zebra": ["cebra"],
    "suitcase": ["maleta", "valija"], "baseball glove": ["guante de béisbol"], "basket": ["canasta", "cesta"],
    "refrigerator": ["refrigerador", "heladera", "nevera"], "ski boot": ["bota de esquí"],
    "tablecloth": ["mantel"], "bracelet": ["pulsera", "brazalete"], "teddy bear": ["oso de peluche"],
    "windshield wiper": ["limpiaparabrisas"],
    "computer mouse": ["mouse de computadora", "ratón de computadora", "mouse", "ratón"], "box": ["caja"],
    "cup": ["taza"], "dress": ["vestido"], "computer monitor": ["monitor de computadora", "monitor"],
    "broccoli": ["brócoli"], "strap": ["correa"], "knob": ["perilla"],
    "plastic bag": ["bolsa de plástico", "bolsa plástica"], "ring": ["anillo"], "awning": ["toldo"],
    "vent": ["rejilla de ventilación", "ventilación"], "banner": ["pancarta", "estandarte"], "cow": ["vaca"],
    "flowerpot": ["maceta"], "tag": ["etiqueta"], "fireplug": ["hidrante", "boca de incendio"], "boot": ["bota"],
}


def build_spanish_names(categories: dict[int, str],
                        names_es: dict[str, list[str]] | None = None) -> dict[int, list[str]]:
    """Map every category_id to its Spanish names; fails fast on missing, unknown or ambiguous entries."""
    names_es = CATEGORY_NAMES_ES if names_es is None else names_es
    known = set(categories.values())
    if set(names_es) != known:
        raise ValueError(f"Spanish names mismatch: missing={sorted(known - set(names_es))}, "
                         f"unknown={sorted(set(names_es) - known)}")
    owners = Counter(term for terms in names_es.values() for term in terms)
    ambiguous = sorted(t for t, n in owners.items() if n > 1)
    if ambiguous:
        raise ValueError(f"Spanish names shared by several categories: {ambiguous}")
    return {cid: list(names_es[name]) for cid, name in sorted(categories.items())}


def size_from_area_ratio(area_ratio: float) -> str:
    if area_ratio < SMALL_MAX_AREA_RATIO:
        return "small"
    if area_ratio < MEDIUM_MAX_AREA_RATIO:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- CAL/VAL split


def load_qa_image_ids(path: Path) -> list[int]:
    with open(path, encoding="utf-8") as f:
        ids = {json.loads(line)["image_id"] for line in f if line.strip()}
    return sorted(ids)


def _allocate_stratified(strata: dict[int, list[int]], n_cal: int, rng: random.Random) -> tuple[list[int], list[int]]:
    """Proportional allocation per stratum (largest remainder) of n_cal items to CAL, rest to VAL."""
    total = sum(len(v) for v in strata.values())
    exact = {k: len(v) * n_cal / total for k, v in strata.items()}
    quota = {k: int(e) for k, e in exact.items()}
    by_remainder = sorted(strata, key=lambda k: (-(exact[k] - quota[k]), k))
    for k in by_remainder[: n_cal - sum(quota.values())]:
        quota[k] += 1
    cal, val = [], []
    for k in sorted(strata):
        members = sorted(strata[k])
        rng.shuffle(members)
        cal.extend(members[: quota[k]])
        val.extend(members[quota[k]:])
    return cal, val


def make_cal_val_split(annotations: dict, qa_image_ids: list[int], seed: int = SEED) -> dict:
    """Reproducible split: QA images 10/10, remaining images stratified by instance-count tercile."""
    rng = random.Random(seed)
    all_ids = sorted(img["id"] for img in annotations["images"])
    qa = sorted(set(qa_image_ids))
    if len(qa) != 2 * N_QA_PER_SPLIT or not set(qa) <= set(all_ids):
        raise ValueError(f"Expected {2 * N_QA_PER_SPLIT} QA images inside the development set, got {len(qa)}")

    counts = Counter(a["image_id"] for a in annotations["annotations"])
    rest = [i for i in all_ids if i not in set(qa)]
    edges = np.quantile([counts[i] for i in rest], [1 / 3, 2 / 3])
    stratum_of = {i: int(np.searchsorted(edges, counts[i], side="right")) for i in rest}

    qa_shuffled = list(qa)
    rng.shuffle(qa_shuffled)
    qa_cal, qa_val = sorted(qa_shuffled[:N_QA_PER_SPLIT]), sorted(qa_shuffled[N_QA_PER_SPLIT:])

    strata: dict[int, list[int]] = {}
    for i, s in stratum_of.items():
        strata.setdefault(s, []).append(i)
    rest_cal, rest_val = _allocate_stratified(strata, N_CAL - N_QA_PER_SPLIT, rng)

    file_names = {img["id"]: img["file_name"] for img in annotations["images"]}
    cal, val = sorted(qa_cal + rest_cal), sorted(qa_val + rest_val)
    return {
        "seed": seed,
        "stratification": "instance-count tercile (non-QA images); QA images split 10/10",
        "tercile_edges": [float(e) for e in edges],
        "cal": cal,
        "val": val,
        "qa_cal": qa_cal,
        "qa_val": qa_val,
        "cal_files": [file_names[i] for i in cal],
        "val_files": [file_names[i] for i in val],
        "strata": {str(k): sorted(v) for k, v in sorted(strata.items())},
    }


def save_split(split: dict, path: Path = SPLIT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return path


def load_split(path: Path = SPLIT_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


_PATHS_FOR_ALIASES = None
try:
    _PATHS_FOR_ALIASES = get_dataset_paths()
except FileNotFoundError:
    pass

CATEGORIES: dict[int, str] = load_categories(_PATHS_FOR_ALIASES.categories) if _PATHS_FOR_ALIASES else {}
LVIS_PROMPT_ALIASES: dict[int, dict] = build_prompt_aliases(CATEGORIES) if CATEGORIES else {}
CATEGORIES_ES: dict[int, list[str]] = build_spanish_names(CATEGORIES) if CATEGORIES else {}
