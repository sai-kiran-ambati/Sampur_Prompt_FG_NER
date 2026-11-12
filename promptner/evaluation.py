"""PromptNER inference and exact-span entity-level evaluation."""

from __future__ import annotations

import json
import random
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .dataset import Entity, Sentence, iter_split, sample_sentences, select_demonstrations
from .prompt import (
    COARSE_SYSTEM_PROMPT,
    FINE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_coarse_user_prompt,
    build_fine_user_prompt,
    build_user_prompt,
    parse_prediction,
)
from .providers import complete, create_client, default_model
from .taxonomy import COARSE_TO_FINE, coarse_type


def entity_metrics(gold: Iterable[tuple[int, int, str]], predicted: Iterable[tuple[int, int, str]]) -> dict[str, float | int]:
    gold_set, predicted_set = set(gold), set(predicted)
    true_positive = len(gold_set & predicted_set)
    false_positive = len(predicted_set - gold_set)
    false_negative = len(gold_set - predicted_set)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "micro_f1": f1,
    }


def evaluate(
    dataset_dir: str | Path,
    provider: str,
    model: str | None = None,
    limit: int = 500,
    seed: int = 42,
    output: str | Path = "outputs/promptner_eval.jsonl",
    workers: int = 4,
    resume: bool = False,
    method: str = "two-stage",
) -> dict:
    if method not in {"one-stage", "two-stage"}:
        raise ValueError("method must be 'one-stage' or 'two-stage'")
    train = iter_split(dataset_dir, "train")
    demonstrations = select_demonstrations(train, count=5)
    test_sample = sample_sentences(iter_split(dataset_dir, "test"), limit=limit, seed=seed)
    client = create_client(provider)
    selected_model = model or default_model(provider)
    if workers < 1:
        raise ValueError("workers must be at least 1")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    records: dict[int, dict] = {}
    if resume and output_path.exists():
        with output_path.open(encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                record = json.loads(line)
                index = int(record["sample_index"])
                if index < 0 or index >= len(test_sample) or tuple(record.get("tokens", ())) != test_sample[index].tokens:
                    raise ValueError("Existing output does not match this seed, sample limit, and dataset")
                if record.get("method", "one-stage") != method:
                    raise ValueError("Existing output was generated with a different method")
                records[index] = record

    # Keep a margin below Sarvam-105B Starter's published 40 requests/minute.
    requests_per_minute = int(os.getenv("SARVAM_RPM", "35")) if provider.lower() == "sarvam" else 0
    if requests_per_minute < 0:
        raise ValueError("SARVAM_RPM must be positive")
    request_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
    rate_lock = threading.Lock()
    next_request_time = [0.0]

    def pace_request() -> None:
        if not request_interval:
            return
        with rate_lock:
            now = time.monotonic()
            wait_for = max(0.0, next_request_time[0] - now)
            if wait_for:
                time.sleep(wait_for)
            next_request_time[0] = time.monotonic() + request_interval

    def predict(sample_index: int, sentence: Sentence) -> tuple[dict, list[tuple[int, int, str]], int]:
        record = {
            "sample_index": sample_index,
            "method": method,
            "tokens": sentence.tokens,
            "text": sentence.text,
            "gold": [asdict(entity) for entity in sentence.entities],
        }
        parse_errors: list[str] = []
        api_errors: list[str] = []
        predictions: tuple[Entity, ...] = ()
        reasoning_effort = "none" if provider.lower() == "sarvam" else None

        def request(system_prompt: str, user_prompt: str) -> str:
            pace_request()
            return complete(
                client,
                selected_model,
                system_prompt,
                user_prompt,
                reasoning_effort=reasoning_effort,
            )

        if method == "one-stage":
            try:
                response = request(SYSTEM_PROMPT, build_user_prompt(sentence, demonstrations))
                parsed = parse_prediction(response, len(sentence.tokens))
                record["raw_response"] = response
                record["explanations"] = parsed.explanations
                parse_errors.extend(parsed.errors)
                predictions = parsed.entities
            except Exception as exc:
                api_errors.append(f"{type(exc).__name__}: {exc}")
        else:
            try:
                coarse_response = request(
                    COARSE_SYSTEM_PROMPT,
                    build_coarse_user_prompt(sentence, demonstrations),
                )
                coarse_parsed = parse_prediction(
                    coarse_response,
                    len(sentence.tokens),
                    allowed_types=COARSE_TO_FINE.keys(),
                )
                record["stage1_raw_response"] = coarse_response
                record["stage1_predictions"] = [asdict(entity) for entity in coarse_parsed.entities]
                parse_errors.extend(f"stage1:{error}" for error in coarse_parsed.errors)
            except Exception as exc:
                coarse_parsed = None
                api_errors.append(f"stage1:{type(exc).__name__}: {exc}")

            if coarse_parsed is not None and coarse_parsed.entities:
                allowed_fine_types = {
                    label
                    for coarse in {entity.label for entity in coarse_parsed.entities}
                    for label in COARSE_TO_FINE[coarse]
                }
                try:
                    fine_response = request(
                        FINE_SYSTEM_PROMPT,
                        build_fine_user_prompt(sentence, coarse_parsed.entities, demonstrations),
                    )
                    fine_parsed = parse_prediction(
                        fine_response,
                        len(sentence.tokens),
                        allowed_types=allowed_fine_types,
                    )
                    record["stage2_raw_response"] = fine_response
                    record["stage2_explanations"] = fine_parsed.explanations
                    parse_errors.extend(f"stage2:{error}" for error in fine_parsed.errors)
                    accepted: list[Entity] = []
                    for entity in fine_parsed.entities:
                        matching_candidate = any(
                            coarse_type(entity.label) == candidate.label
                            and candidate.start <= entity.start
                            and entity.end <= candidate.end
                            for candidate in coarse_parsed.entities
                        )
                        if matching_candidate:
                            accepted.append(entity)
                        else:
                            parse_errors.append(
                                f"stage2:entity_outside_coarse_proposals:{entity.start}:{entity.end}:{entity.label}"
                            )
                    predictions = tuple(accepted)
                except Exception as exc:
                    api_errors.append(f"stage2:{type(exc).__name__}: {exc}")

        record["predictions"] = [asdict(entity) for entity in predictions]
        record["parse_errors"] = parse_errors
        if api_errors:
            record["api_error"] = " | ".join(api_errors)
        return record, [], 1 if api_errors or parse_errors else 0

    pending_indices = [
        index for index in range(len(test_sample))
        if index not in records or (resume and "api_error" in records[index])
    ]

    def checkpoint(handle) -> None:
        handle.seek(0)
        handle.truncate()
        for index in sorted(records):
            handle.write(json.dumps(records[index], ensure_ascii=False) + "\n")
        handle.flush()

    with output_path.open("w+", encoding="utf-8") as handle:
        if records:
            checkpoint(handle)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(predict, index, test_sample[index])
                for index in pending_indices
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                record, _, _ = future.result()
                records[record["sample_index"]] = record
                if completed % 10 == 0 or completed == len(pending_indices):
                    checkpoint(handle)
                if completed % 25 == 0 or completed == len(pending_indices):
                    print(f"Processed {completed}/{len(pending_indices)} pending test sentences", file=sys.stderr, flush=True)

    gold = [
        (entity.start, entity.end, entity.label)
        for sentence in test_sample for entity in sentence.entities
    ]
    predicted = [
        (int(entity["start"]), int(entity["end"]), entity["label"])
        for index in range(len(test_sample))
        for entity in records.get(index, {}).get("predictions", [])
    ]
    api_failures = sum(1 for record in records.values() if record.get("api_error"))
    parse_failures = sum(1 for record in records.values() if record.get("parse_errors"))
    metrics = entity_metrics(gold, predicted)
    result = {
        "provider": provider,
        "model": selected_model,
        "method": method,
        "split": "test",
        "sample_size": len(test_sample),
        "seed": seed,
        "demonstrations": 5,
        "workers": workers,
        "resumed_records": len(records) - len(pending_indices),
        "api_failures": api_failures,
        "parse_failures": parse_failures,
        "parse_issues": sum(len(record.get("parse_errors", [])) for record in records.values()),
        "elapsed_seconds": round(time.monotonic() - start_time, 2),
        "metrics": metrics,
        "predictions_file": str(output_path),
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    result["summary_file"] = str(summary_path)
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
