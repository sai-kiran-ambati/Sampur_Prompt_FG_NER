import json
import unittest

from promptner.dataset import Entity, Sentence, parse_bio_tokens
from promptner.evaluation import entity_metrics
from promptner.prompt import parse_prediction
from promptner.taxonomy import TYPE_DEFINITIONS


class BioParsingTests(unittest.TestCase):
    def test_parses_multitoken_entity_and_adjacent_entities(self):
        self.assertEqual(
            parse_bio_tokens(["A", "B", "C", "D"], ["B-person-actor", "I-person-actor", "B-location-GPE", "O"]),
            (Entity(0, 2, "person-actor"), Entity(2, 3, "location-GPE")),
        )

    def test_orphan_inside_tag_starts_entity_and_outside_is_ignored(self):
        self.assertEqual(parse_bio_tokens(["x", "y"], ["I-product-car", "O"]), (Entity(0, 1, "product-car"),))

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            parse_bio_tokens(["x"], [])


class PredictionParsingTests(unittest.TestCase):
    def test_parses_json_and_drops_duplicate_unknown_or_invalid_spans(self):
        payload = {
            "entities": [
                {"start_token": 0, "end_token": 2, "type": "person-actor", "explanation": "actor"},
                {"start_token": 0, "end_token": 2, "type": "person-actor"},
                {"start_token": 1, "end_token": 2, "type": "invented-type"},
                {"start_token": 4, "end_token": 5, "type": "person-actor"},
            ]
        }
        result = parse_prediction(json.dumps(payload), token_count=3)
        self.assertEqual(result.entities, (Entity(0, 2, "person-actor"),))
        self.assertEqual(len(result.errors), 3)

    def test_malformed_response_is_reported(self):
        result = parse_prediction("not json", token_count=2)
        self.assertFalse(result.entities)
        self.assertTrue(result.errors)

    def test_code_fenced_json_is_accepted(self):
        result = parse_prediction('```json\n{"entities": []}\n```', token_count=2)
        self.assertFalse(result.errors)

    def test_repeat_mentions_keep_distinct_offsets(self):
        payload = {"entities": [
            {"start_token": 0, "end_token": 1, "type": "person-actor"},
            {"start_token": 3, "end_token": 4, "type": "person-actor"},
        ]}
        result = parse_prediction(json.dumps(payload), token_count=4)
        self.assertEqual(len(result.entities), 2)


class MetricTests(unittest.TestCase):
    def test_entity_micro_metrics_use_exact_span_and_type(self):
        gold = [(0, 2, "person-actor"), (3, 4, "location-GPE")]
        predicted = [(0, 2, "person-actor"), (3, 4, "location-other")]
        metrics = entity_metrics(gold, predicted)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["micro_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
