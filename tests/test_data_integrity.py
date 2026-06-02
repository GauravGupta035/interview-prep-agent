"""
test_data_integrity.py — Schema checks on the JSON question banks.

These run against the real data files, so a refresh commit (or a hand edit)
that writes a malformed entry or a duplicate ID fails CI.
"""

import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data"

CS_REQUIRED = {
    "id",
    "topic",
    "category",
    "difficulty",
    "question_types",
    "key_points",
    "common_mistakes",
}
BEH_REQUIRED = {"id", "competency", "question", "what_they_want", "red_flags"}
VALID_CATEGORIES = {
    "networking",
    "systems",
    "databases",
    "software_engineering",
    "security",
    "cloud",
}


def _load(name):
    return json.loads((DATA / name).read_text())


class TestCsConcepts:
    def test_is_a_nonempty_list(self):
        data = _load("cs_concepts.json")
        assert isinstance(data, list) and data

    def test_every_entry_has_required_keys(self):
        for entry in _load("cs_concepts.json"):
            missing = CS_REQUIRED - entry.keys()
            assert not missing, f"{entry.get('id')} missing {missing}"

    def test_ids_are_unique(self):
        ids = [e["id"] for e in _load("cs_concepts.json")]
        assert len(ids) == len(set(ids))

    def test_categories_are_valid(self):
        for entry in _load("cs_concepts.json"):
            assert entry["category"] in VALID_CATEGORIES

    def test_question_types_well_formed(self):
        for entry in _load("cs_concepts.json"):
            assert entry["question_types"], f"{entry['id']} has no question_types"
            for qt in entry["question_types"]:
                assert "type" in qt and "q" in qt


class TestBehaviouralQuestions:
    def test_is_a_nonempty_list(self):
        data = _load("behavioural_questions.json")
        assert isinstance(data, list) and data

    def test_every_entry_has_required_keys(self):
        for entry in _load("behavioural_questions.json"):
            missing = BEH_REQUIRED - entry.keys()
            assert not missing, f"{entry.get('id')} missing {missing}"

    def test_ids_are_unique(self):
        ids = [e["id"] for e in _load("behavioural_questions.json")]
        assert len(ids) == len(set(ids))

    def test_questions_are_nonempty_strings(self):
        for entry in _load("behavioural_questions.json"):
            assert isinstance(entry["question"], str) and entry["question"].strip()
