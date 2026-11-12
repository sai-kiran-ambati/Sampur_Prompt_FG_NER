"""Prompt construction and resilient JSON parsing for PromptNER."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .dataset import Entity, Sentence
from .taxonomy import COARSE_DEFINITIONS, COARSE_TO_FINE, TYPE_DEFINITIONS, coarse_type

SYSTEM_PROMPT = """You perform fine-grained named entity recognition on Telugu text.
Follow the supplied entity definitions. Consider an entity valid only when it fits
one of those definitions. Do not translate or normalize the Telugu text. Use only
the exact fine-grained type identifiers supplied. Return JSON only, with this
shape: {\"entities\":[{\"start_token\":0,\"end_token\":1,\"type\":\"person-actor\",\"explanation\":\"brief reason\"}]}
Token positions are zero-based; start_token is inclusive and end_token exclusive.
Each entry must cover one distinct contiguous entity span. Return an empty entities
array when there are no entities. Never include an entity solely because it is a
common noun or because the sentence contains a capitalized word."""

COARSE_SYSTEM_PROMPT = """You perform coarse-grained named entity recognition on Telugu text.
Identify named entity spans and classify each as exactly one of the supplied coarse
classes: art, building, event, location, organization, other, person, or product.
Do not guess a fine-grained subtype at this stage. Do not translate or normalize
the Telugu text. Return JSON only with shape {"entities":[{"start_token":0,
"end_token":1,"type":"person","explanation":"brief reason"}]}. Token indexes
are zero-based; start_token is inclusive and end_token exclusive. Return an empty
entities array when no named entities are present."""

FINE_SYSTEM_PROMPT = """You perform the second stage of a two-stage Telugu NER system.
The first stage proposed coarse entity spans. For each proposal, decide whether it
is a valid named entity and assign exactly one fine-grained subtype from the
provided options for that proposal's coarse class. You may tighten span boundaries
to the exact entity mention, but the returned span must be inside a proposed span
and its type must belong to that proposal's coarse class. Reject false coarse-stage
proposals by omitting them. Do not invent entities outside the proposals. Return
JSON only with shape {"entities":[{"start_token":0,"end_token":1,
"type":"person-actor","explanation":"brief reason"}]}. Token indexes are
zero-based; start_token is inclusive and end_token exclusive."""


@dataclass(frozen=True)
class ParsedPrediction:
    entities: tuple[Entity, ...]
    explanations: tuple[str, ...]
    errors: tuple[str, ...]


def _format_entities(entities: tuple[Entity, ...], tokens: tuple[str, ...]) -> str:
    values = []
    for entity in entities:
        values.append({
            "start_token": entity.start,
            "end_token": entity.end,
            "mention": " ".join(tokens[entity.start:entity.end]),
            "type": entity.label,
            "explanation": f"The mention matches the definition of {entity.label}.",
        })
    return json.dumps({"entities": values}, ensure_ascii=False)


def build_user_prompt(sentence: Sentence, demonstrations: list[Sentence]) -> str:
    definitions = "\n".join(
        f"- {label}: {definition}" for label, definition in TYPE_DEFINITIONS.items()
    )
    examples = []
    for index, example in enumerate(demonstrations, start=1):
        indexed = "\n".join(f"[{i}] {token}" for i, token in enumerate(example.tokens))
        examples.append(
            f"Example {index}\nTokens:\n{indexed}\nGold JSON:\n"
            f"{_format_entities(example.entities, example.tokens)}"
        )
    indexed_sentence = "\n".join(f"[{i}] {token}" for i, token in enumerate(sentence.tokens))
    examples_text = "\n\n".join(examples)
    return (
        "Entity type definitions (all IDs are exact labels):\n"
        f"{definitions}\n\n"
        "Labeled Telugu examples:\n"
        f"{examples_text}\n\n"
        "Identify all entities in this Telugu sentence. Return the required JSON.\n"
        f"Tokens:\n{indexed_sentence}"
    )


def _extract_json(text: str) -> object:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise
        value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        return value


def parse_prediction(text: str, token_count: int, allowed_types=None) -> ParsedPrediction:
    errors: list[str] = []
    try:
        payload = _extract_json(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return ParsedPrediction((), (), (f"invalid_json: {exc}",))
    if not isinstance(payload, dict) or not isinstance(payload.get("entities"), list):
        return ParsedPrediction((), (), ("missing_entities_array",))

    entities: list[Entity] = []
    explanations: list[str] = []
    seen: set[tuple[int, int, str]] = set()
    valid_types = set(TYPE_DEFINITIONS if allowed_types is None else allowed_types)
    for index, item in enumerate(payload["entities"]):
        if not isinstance(item, dict):
            errors.append(f"entity_{index}: expected_object")
            continue
        try:
            start, end = int(item["start_token"]), int(item["end_token"])
            label = str(item["type"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"entity_{index}: missing_or_invalid_fields")
            continue
        if label not in valid_types:
            errors.append(f"entity_{index}: unknown_type:{label}")
            continue
        if start < 0 or end <= start or end > token_count:
            errors.append(f"entity_{index}: invalid_span:{start}:{end}")
            continue
        key = (start, end, label)
        if key in seen:
            errors.append(f"entity_{index}: duplicate")
            continue
        seen.add(key)
        entities.append(Entity(start, end, label))
        explanations.append(str(item.get("explanation", "")))
    return ParsedPrediction(tuple(entities), tuple(explanations), tuple(errors))


def build_coarse_user_prompt(sentence: Sentence, demonstrations: list[Sentence]) -> str:
    coarse_demonstrations = []
    for example in demonstrations:
        coarse_entities = tuple(
            Entity(entity.start, entity.end, coarse_type(entity.label)) for entity in example.entities
        )
        coarse_example = Sentence(example.tokens, example.tags, coarse_entities)
        coarse_demonstrations.append(coarse_example)
    definitions = "\n".join(
        f"- {label}: {definition}" for label, definition in COARSE_DEFINITIONS.items()
    )
    examples = []
    for index, example in enumerate(coarse_demonstrations, start=1):
        indexed = "\n".join(f"[{i}] {token}" for i, token in enumerate(example.tokens))
        examples.append(
            f"Example {index}\nTokens:\n{indexed}\nGold JSON:\n"
            f"{_format_entities(example.entities, example.tokens)}"
        )
    indexed_sentence = "\n".join(f"[{i}] {token}" for i, token in enumerate(sentence.tokens))
    examples_text = "\n\n".join(examples)
    return (
        "Coarse entity class definitions:\n"
        f"{definitions}\n\nLabeled Telugu examples:\n"
        f"{examples_text}\n\n"
        "Find all named entities and return their coarse classes.\n"
        f"Tokens:\n{indexed_sentence}"
    )


def build_fine_user_prompt(
    sentence: Sentence,
    coarse_entities: tuple[Entity, ...],
    demonstrations: list[Sentence],
) -> str:
    candidate_classes = {entity.label for entity in coarse_entities}
    candidate_lines = []
    for candidate_id, entity in enumerate(coarse_entities):
        mention = " ".join(sentence.tokens[entity.start:entity.end])
        candidate_lines.append(
            f"- candidate_id={candidate_id}, span=[{entity.start},{entity.end}), "
            f"coarse_type={entity.label}, mention={json.dumps(mention, ensure_ascii=False)}"
        )

    relevant_types = [
        label for coarse in COARSE_DEFINITIONS if coarse in candidate_classes
        for label in COARSE_TO_FINE[coarse]
    ]
    definitions = "\n".join(f"- {label}: {TYPE_DEFINITIONS[label]}" for label in relevant_types)
    examples = []
    for example in demonstrations:
        filtered_entities = tuple(
            entity for entity in example.entities if coarse_type(entity.label) in candidate_classes
        )
        if filtered_entities:
            examples.append(Sentence(example.tokens, example.tags, filtered_entities))
    example_blocks = []
    for index, example in enumerate(examples, start=1):
        indexed = "\n".join(f"[{i}] {token}" for i, token in enumerate(example.tokens))
        example_blocks.append(
            f"Example {index}\nTokens:\n{indexed}\nGold fine-grained JSON:\n"
            f"{_format_entities(example.entities, example.tokens)}"
        )
    indexed_sentence = "\n".join(f"[{i}] {token}" for i, token in enumerate(sentence.tokens))
    candidates_text = "\n".join(candidate_lines)
    examples_text = "\n\n".join(example_blocks) if example_blocks else "(none)"
    return (
        "Fine-grained options (choose only among these exact labels):\n"
        f"{definitions}\n\n"
        "First-stage coarse entity proposals to refine:\n"
        f"{candidates_text}\n\n"
        "Relevant labeled Telugu examples:\n"
        f"{examples_text}\n\n"
        "Assign a fine-grained type to each valid proposal. Reject false proposals.\n"
        f"Sentence tokens:\n{indexed_sentence}"
    )
