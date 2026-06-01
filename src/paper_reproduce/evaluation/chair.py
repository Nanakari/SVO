"""COCO/CHAIR caption hallucination metrics.

The default scorer below mirrors the original CHAIR evaluator's core behavior without vendoring
the upstream Python 2 code: normalize generated caption object mentions through a COCO synonym map,
compare them with image-level COCO objects, and report CHAIRs/CHAIRi. The previous project-local
extractor-based implementation is retained as the explicit internal fallback.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from paper_reproduce.evaluation.common import safe_divide, token_length
from paper_reproduce.extraction import ObjectExtractor


class OfficialChairMapper:
    """Official-CHAIR-compatible COCO object mention normalizer."""

    def __init__(self, categories: Iterable[str]) -> None:
        self.inverse_synonym_dict: dict[str, str] = {}
        for category in categories:
            normalized = _normalize_phrase(category)
            self.inverse_synonym_dict[normalized] = normalized
        for canonical, aliases in _OFFICIAL_COMPAT_SYNONYMS.items():
            self.inverse_synonym_dict[canonical] = canonical
            for alias in aliases:
                self.inverse_synonym_dict[_normalize_phrase(alias)] = canonical

        self.double_word_dict = {phrase: phrase for phrase in _COCO_DOUBLE_WORDS}
        for animal_word in _ANIMAL_WORDS:
            self.double_word_dict[f"baby {animal_word}"] = animal_word
            self.double_word_dict[f"adult {animal_word}"] = animal_word
        for vehicle_word in _VEHICLE_WORDS:
            self.double_word_dict[f"passenger {vehicle_word}"] = vehicle_word
        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict["wine glas"] = "wine glass"

    def canonicalize_phrase(self, value: str) -> str | None:
        """Return the official-compatible COCO object name for a category or caption phrase."""

        normalized = _normalize_phrase(value)
        if normalized in self.inverse_synonym_dict:
            return self.inverse_synonym_dict[normalized]
        double_word = self.double_word_dict.get(normalized)
        if double_word is not None:
            return self.inverse_synonym_dict.get(double_word, double_word)
        return None

    def caption_to_objects(self, caption: str) -> tuple[list[str], list[str], list[int], list[str]]:
        words = [_simple_singular(token) for token in _tokenize(caption.lower())]

        merged: list[str] = []
        original_indices: list[int] = []
        index = 0
        while index < len(words):
            original_indices.append(index)
            double_word = " ".join(words[index : index + 2])
            if double_word in self.double_word_dict:
                merged.append(self.double_word_dict[double_word])
                index += 2
            else:
                merged.append(words[index])
                index += 1

        if "toilet" in merged and "seat" in merged:
            merged = [word for word in merged if word != "seat"]

        raw_words: list[str] = []
        node_words: list[str] = []
        mention_indices: list[int] = []
        for index, word in enumerate(merged):
            canonical = self.inverse_synonym_dict.get(word)
            if canonical is None:
                continue
            raw_words.append(word)
            node_words.append(canonical)
            if index < len(original_indices):
                mention_indices.append(original_indices[index])
        return raw_words, node_words, mention_indices, merged


def evaluate_chair_records_official(
    records: list[Mapping[str, Any]],
    *,
    gt_by_image: Mapping[str, set[str]],
    mapper: OfficialChairMapper,
    text_field: str = "caption",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute official-compatible CHAIR metrics."""

    evaluated_images = 0
    missing_gt = 0
    hallucinated_sentences = 0
    total_mentions = 0
    hallucinated_mentions = 0
    total_length = 0
    coverage_sum = 0.0
    coverage_denominator = 0

    for record in records:
        image_id = str(record.get("image_id", ""))
        gt_objects = gt_by_image.get(image_id)
        if gt_objects is None:
            missing_gt += 1
            continue
        text = str(record.get(text_field) or record.get("revised_caption") or record.get("caption") or "")
        raw_words, mention_names, _, _ = mapper.caption_to_objects(text)
        hallucinated = [name for name in mention_names if name not in gt_objects]

        evaluated_images += 1
        total_length += token_length(text)
        total_mentions += len(raw_words)
        hallucinated_mentions += len(hallucinated)
        hallucinated_sentences += int(bool(hallucinated))
        if gt_objects:
            coverage_sum += len(set(mention_names).intersection(gt_objects)) / len(gt_objects)
            coverage_denominator += 1

    metrics = {
        "chairs": safe_divide(hallucinated_sentences, evaluated_images),
        "chairi": safe_divide(hallucinated_mentions, total_mentions),
        "average_length": safe_divide(total_length, evaluated_images),
        "correct_object_coverage": safe_divide(coverage_sum, coverage_denominator),
    }
    counts = {
        "records": len(records),
        "evaluated_images": evaluated_images,
        "missing_gt": missing_gt,
        "hallucinated_sentences": hallucinated_sentences,
        "object_mentions": total_mentions,
        "hallucinated_object_mentions": hallucinated_mentions,
        "coverage_images": coverage_denominator,
        "chair_backend": "official",
    }
    return metrics, counts


def evaluate_chair_records_internal(
    records: list[Mapping[str, Any]],
    *,
    gt_by_image: Mapping[str, set[str]],
    extractor: ObjectExtractor,
    text_field: str = "caption",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute CHAIRs, CHAIRi, Average Length, and Correct Object Coverage."""

    evaluated_images = 0
    missing_gt = 0
    hallucinated_sentences = 0
    total_mentions = 0
    hallucinated_mentions = 0
    total_length = 0
    coverage_sum = 0.0
    coverage_denominator = 0

    for record in records:
        image_id = str(record.get("image_id", ""))
        gt_objects = gt_by_image.get(image_id)
        if gt_objects is None:
            missing_gt += 1
            continue
        text = str(record.get(text_field) or record.get("revised_caption") or record.get("caption") or "")
        mentions = extractor.extract(text)
        mention_names = [mention.normalized for mention in mentions]
        hallucinated = [name for name in mention_names if name not in gt_objects]

        evaluated_images += 1
        total_length += token_length(text)
        total_mentions += len(mention_names)
        hallucinated_mentions += len(hallucinated)
        hallucinated_sentences += int(bool(hallucinated))
        if gt_objects:
            coverage_sum += len(set(mention_names).intersection(gt_objects)) / len(gt_objects)
            coverage_denominator += 1

    metrics = {
        "chairs": safe_divide(hallucinated_sentences, evaluated_images),
        "chairi": safe_divide(hallucinated_mentions, total_mentions),
        "average_length": safe_divide(total_length, evaluated_images),
        "correct_object_coverage": safe_divide(coverage_sum, coverage_denominator),
    }
    counts = {
        "records": len(records),
        "evaluated_images": evaluated_images,
        "missing_gt": missing_gt,
        "hallucinated_sentences": hallucinated_sentences,
        "object_mentions": total_mentions,
        "hallucinated_object_mentions": hallucinated_mentions,
        "coverage_images": coverage_denominator,
        "chair_backend": "internal",
    }
    return metrics, counts


# Backward-compatible legacy alias. CLI/paper metrics choose the backend explicitly.
evaluate_chair_records = evaluate_chair_records_internal


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def _normalize_phrase(value: str) -> str:
    return " ".join(_simple_singular(token) for token in _tokenize(value.lower()))


def _simple_singular(word: str) -> str:
    if word in {"skis", "scissors", "sports"}:
        return word
    irregular = {
        "people": "person",
        "children": "child",
        "men": "man",
        "women": "woman",
        "oxen": "ox",
        "geese": "goose",
        "knives": "knife",
    }
    if word in irregular:
        return irregular[word]
    if len(word) > 4 and word.endswith("ies"):
        return f"{word[:-3]}y"
    if len(word) > 4 and word.endswith("ves"):
        return f"{word[:-3]}f"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes", "zes", "ses")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


_COCO_DOUBLE_WORDS = {
    "motor bike",
    "motor cycle",
    "air plane",
    "traffic light",
    "street light",
    "traffic signal",
    "stop light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "suit case",
    "sports ball",
    "baseball bat",
    "baseball glove",
    "tennis racket",
    "wine glass",
    "hot dog",
    "cell phone",
    "mobile phone",
    "teddy bear",
    "hair drier",
    "potted plant",
    "bow tie",
    "laptop computer",
    "stove top oven",
    "home plate",
    "train track",
}

_ANIMAL_WORDS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal", "cub"}
_VEHICLE_WORDS = {"jet", "train"}

_OFFICIAL_COMPAT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "person": (
        "girl",
        "boy",
        "man",
        "woman",
        "kid",
        "baby",
        "child",
        "chef",
        "baker",
        "people",
        "adult",
        "rider",
        "children",
        "worker",
        "passenger",
        "policeman",
        "cop",
        "officer",
        "lady",
        "cowboy",
        "bride",
        "groom",
        "male",
        "female",
        "guy",
        "traveler",
        "mother",
        "father",
        "gentleman",
        "player",
        "skier",
        "snowboarder",
        "skater",
        "skateboarder",
        "student",
        "driver",
    ),
    "bicycle": ("bike", "unicycle", "minibike", "trike"),
    "car": ("automobile", "van", "minivan", "sedan", "suv", "hatchback", "cab", "jeep", "taxi"),
    "motorcycle": ("scooter", "motor bike", "motor cycle", "motorbike", "moped"),
    "airplane": ("jetliner", "plane", "air plane", "aircraft", "jet", "airbus", "biplane", "seaplane"),
    "bus": ("minibus", "trolley"),
    "train": ("locomotive", "tramway", "caboose"),
    "truck": ("pickup", "lorry", "hauler", "firetruck"),
    "boat": ("ship", "liner", "sailboat", "motorboat", "dinghy", "yacht", "kayak", "canoe", "ferry"),
    "traffic light": ("street light", "traffic signal", "stop light", "streetlight", "stoplight"),
    "fire hydrant": ("hydrant",),
    "stop sign": (),
    "parking meter": (),
    "bench": ("pew",),
    "bird": ("ostrich", "owl", "seagull", "goose", "duck", "parrot", "swan", "turkey", "pigeon"),
    "cat": ("kitten", "feline", "tabby"),
    "dog": ("puppy", "beagle", "pup", "canine", "terrier", "poodle", "labrador", "doggie", "doggy", "hound"),
    "horse": ("colt", "pony", "racehorse", "stallion", "equine", "mare", "foal"),
    "sheep": ("lamb", "ram", "goat", "ewe"),
    "cow": ("cattle", "oxen", "ox", "calf", "bull", "buffalo", "bison"),
    "elephant": (),
    "bear": ("panda",),
    "zebra": (),
    "giraffe": (),
    "backpack": ("knapsack",),
    "umbrella": (),
    "handbag": ("wallet", "purse", "briefcase"),
    "tie": ("bow", "bow tie"),
    "suitcase": ("suit case", "luggage"),
    "frisbee": (),
    "skis": ("ski",),
    "snowboard": (),
    "sports ball": ("ball",),
    "kite": (),
    "baseball bat": (),
    "baseball glove": (),
    "skateboard": (),
    "surfboard": ("longboard", "skimboard", "shortboard", "wakeboard"),
    "tennis racket": ("racket",),
    "bottle": (),
    "wine glass": (),
    "cup": (),
    "fork": (),
    "knife": ("pocketknife", "knive"),
    "spoon": (),
    "bowl": ("container",),
    "banana": (),
    "apple": (),
    "sandwich": ("burger", "sub", "cheeseburger", "hamburger"),
    "orange": (),
    "broccoli": (),
    "carrot": (),
    "hot dog": (),
    "pizza": (),
    "donut": ("doughnut", "bagel"),
    "cake": ("cheesecake", "cupcake", "shortcake", "pancake"),
    "chair": ("seat", "stool"),
    "couch": ("sofa", "recliner", "futon", "loveseat", "settee"),
    "potted plant": ("houseplant",),
    "bed": (),
    "dining table": ("table", "desk"),
    "toilet": ("urinal", "commode", "lavatory", "potty"),
    "tv": ("monitor", "television", "televison"),
    "laptop": ("computer", "notebook", "netbook", "laptop computer"),
    "mouse": (),
    "remote": (),
    "keyboard": (),
    "cell phone": ("mobile phone", "phone", "cellphone", "telephone", "smartphone", "iphone"),
    "microwave": (),
    "oven": ("stovetop", "stove", "stove top oven"),
    "toaster": (),
    "sink": (),
    "refrigerator": ("fridge", "freezer"),
    "book": (),
    "clock": (),
    "vase": (),
    "scissors": (),
    "teddy bear": ("teddybear",),
    "hair drier": ("hairdryer",),
    "toothbrush": (),
}
