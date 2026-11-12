"""Load the token-per-row SampurNER text dataset and reconstruct BIO spans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class Entity:
    start: int  # inclusive token index
    end: int  # exclusive token index
    label: str  # fine-grained label without the BIO prefix


@dataclass(frozen=True)
class Sentence:
    tokens: tuple[str, ...]
    tags: tuple[str, ...]
    entities: tuple[Entity, ...]

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


def parse_bio_tokens(tokens: Sequence[str], tags: Sequence[str]) -> tuple[Entity, ...]:
    """Convert BIO token labels into half-open spans, tolerating orphan I tags."""
    if len(tokens) != len(tags):
        raise ValueError(f"Token/tag length mismatch: {len(tokens)} != {len(tags)}")

    entities: list[Entity] = []
    start: int | None = None
    active_label: str | None = None

    def close(end: int) -> None:
        nonlocal start, active_label
        if start is not None and active_label is not None:
            entities.append(Entity(start=start, end=end, label=active_label))
        start = None
        active_label = None

    for index, tag in enumerate(tags):
        if tag == "O" or "-" not in tag:
            close(index)
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B":
            close(index)
            start, active_label = index, label
        elif prefix == "I":
            if start is None or active_label != label:
                close(index)
                start, active_label = index, label
        else:
            close(index)
    close(len(tokens))
    return tuple(entities)


def _sentences_from_lines(lines: Iterable[str]) -> Iterator[Sentence]:
    tokens: list[str] = []
    tags: list[str] = []

    def build() -> Sentence | None:
        if not tokens:
            return None
        return Sentence(tuple(tokens), tuple(tags), parse_bio_tokens(tokens, tags))

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            sentence = build()
            if sentence is not None:
                yield sentence
            tokens.clear()
            tags.clear()
            continue
        if "\t" in line:
            token, tag = line.rsplit("\t", 1)
            token, tag = token.strip(), tag.strip()
        else:
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Expected token and BIO tag in dataset row: {raw_line!r}")
            token, tag = " ".join(fields[:-1]), fields[-1]
        # The released text conversion contains occasional empty-token O rows.
        if not token:
            if tag == "O":
                continue
            raise ValueError(f"Expected token and BIO tag in dataset row: {raw_line!r}")
        tokens.append(token)
        tags.append(tag)
    sentence = build()
    if sentence is not None:
        yield sentence


def iter_split(dataset_dir: str | Path, split: str) -> Iterator[Sentence]:
    """Yield sentences from a Hugging Face Dataset saved via save_to_disk."""
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_dir))
    if split not in dataset:
        raise ValueError(f"Split {split!r} not present in {dataset_dir}")
    if "text" not in dataset[split].column_names:
        raise ValueError(f"Expected a 'text' column in split {split!r}")
    yield from _sentences_from_lines(row["text"] for row in dataset[split])


def sample_sentences(sentences: Iterable[Sentence], limit: int, seed: int = 42) -> list[Sentence]:
    """Deterministic reservoir sample without materializing the entire split."""
    import random

    if limit < 1:
        raise ValueError("Sample limit must be positive")
    rng = random.Random(seed)
    reservoir: list[Sentence] = []
    for index, sentence in enumerate(sentences):
        if index < limit:
            reservoir.append(sentence)
        else:
            replacement = rng.randrange(index + 1)
            if replacement < limit:
                reservoir[replacement] = sentence
    return reservoir


def select_demonstrations(sentences: Iterable[Sentence], count: int = 5, seed: int = 17) -> list[Sentence]:
    """Select reproducible, label-diverse demonstrations from the training split."""
    import random

    rng = random.Random(seed)
    pool: list[Sentence] = []
    seen = 0
    for sentence in sentences:
        if sentence.entities:
            seen += 1
            if len(pool) < 4000:
                pool.append(sentence)
            else:
                replacement = rng.randrange(seen)
                if replacement < 4000:
                    pool[replacement] = sentence
    if not pool:
        raise ValueError("No labeled training sentences available for demonstrations")

    selected: list[Sentence] = []
    uncovered: set[str] = set()
    remaining = list(pool)
    # Greedy set cover favors examples with several new entity types.
    for _ in range(min(count, len(remaining))):
        best_index = max(
            range(len(remaining)),
            key=lambda i: (len({entity.label for entity in remaining[i].entities} - uncovered), -i),
        )
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        uncovered.update(entity.label for entity in chosen.entities)
    return selected
