"""
test_topic_refresher.py — Unit tests for the web-search refresh logic.

All Tavily and LLM calls are mocked. Covers deduplication, ID assignment,
LLM-output parse resilience, and the two end-to-end refresh entry points.

test_refresh_behavioural_drops_duplicates is the regression guard for Bug B:
refresh_behavioural_bank used to assign IDs to the raw (un-deduplicated) list,
silently re-adding duplicate questions to the bank.
"""

from unittest.mock import MagicMock, patch

from data import topic_refresher as tr


# ── deduplicate_concepts ─────────────────────────────────────────────────────


class TestDeduplicateConcepts:
    def test_removes_case_insensitive_duplicate(self):
        existing = [{"topic": "TCP vs UDP"}]
        new = [{"topic": "tcp vs udp"}, {"topic": "DNS"}]
        result = tr.deduplicate_concepts(existing, new)
        assert [t["topic"] for t in result] == ["DNS"]

    def test_removes_duplicates_within_same_batch(self):
        result = tr.deduplicate_concepts([], [{"topic": "DNS"}, {"topic": "dns"}])
        assert len(result) == 1

    def test_skips_entries_with_no_topic(self):
        result = tr.deduplicate_concepts([], [{"topic": ""}, {"foo": "bar"}])
        assert result == []


# ── assign_concept_ids ───────────────────────────────────────────────────────


class TestAssignConceptIds:
    def test_continues_from_highest_per_category(self):
        existing = [{"id": "networking_005", "category": "networking"}]
        new = [{"category": "networking"}, {"category": "networking"}]
        result = tr.assign_concept_ids(existing, new)
        assert result[0]["id"] == "networking_006"
        assert result[1]["id"] == "networking_007"

    def test_starts_at_one_for_unseen_category(self):
        result = tr.assign_concept_ids([], [{"category": "cloud"}])
        assert result[0]["id"] == "cloud_001"

    def test_ignores_malformed_existing_ids(self):
        existing = [{"id": "weird-id", "category": "systems"}]
        result = tr.assign_concept_ids(existing, [{"category": "systems"}])
        assert result[0]["id"] == "systems_001"


# ── deduplicate_behavioural ──────────────────────────────────────────────────


class TestDeduplicateBehavioural:
    def test_dedupes_on_60_char_fingerprint(self):
        existing = [{"question": "Tell me about a time you solved a hard problem."}]
        new = [
            {"question": "Tell me about a time you solved a hard problem."},
            {"question": "Describe a situation where you led a team."},
        ]
        result = tr.deduplicate_behavioural(existing, new)
        assert len(result) == 1
        assert result[0]["question"].startswith("Describe")

    def test_questions_sharing_first_60_chars_are_duplicates(self):
        base = "Tell me about a time you had to deal with a really difficult"
        existing = [{"question": base + " stakeholder."}]
        new = [{"question": base + " teammate."}]  # same first 60 chars
        assert tr.deduplicate_behavioural(existing, new) == []


# ── assign_behavioural_ids ───────────────────────────────────────────────────


class TestAssignBehaviouralIds:
    def test_sequential_from_highest(self):
        existing = [{"id": "beh_003"}]
        new = [{"question": "q1"}, {"question": "q2"}]
        result = tr.assign_behavioural_ids(existing, new)
        assert result[0]["id"] == "beh_004"
        assert result[1]["id"] == "beh_005"


# ── extract parsing resilience ───────────────────────────────────────────────


class TestExtractParsing:
    @patch.object(tr, "llm")
    def test_strips_json_fences(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content='```json\n[{"topic": "X"}]\n```'
        )
        result = tr.extract__topics_from_search("raw", [])
        assert result == [{"topic": "X"}]

    @patch.object(tr, "llm")
    def test_invalid_json_returns_empty(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="not json {")
        assert tr.extract__topics_from_search("raw", []) == []

    @patch.object(tr, "llm")
    def test_non_array_returns_empty(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content='{"topic": "X"}')
        assert tr.extract__topics_from_search("raw", []) == []


# ── refresh_topic_bank (CS) ──────────────────────────────────────────────────


class TestRefreshTopicBank:
    @patch.object(tr, "save_concepts")
    @patch.object(tr, "extract__topics_from_search")
    @patch.object(tr, "search_for_topics")
    @patch.object(tr, "load_existing_concepts")
    def test_adds_only_unique_topics(self, mock_load, mock_search, mock_extract, mock_save):
        mock_load.return_value = [{"id": "networking_001", "topic": "TCP vs UDP", "category": "networking"}]
        mock_search.return_value = "raw text"
        mock_extract.return_value = [
            {"topic": "TCP vs UDP", "category": "networking"},  # duplicate
            {"topic": "DNS", "category": "networking"},  # new
        ]
        result = tr.refresh_topic_bank()
        assert result["added"] == 1
        assert result["new_topics"] == ["DNS"]
        assert result["error"] is None
        mock_save.assert_called_once()

    @patch.object(tr, "search_for_topics", return_value="")
    @patch.object(tr, "load_existing_concepts", return_value=[])
    def test_empty_search_returns_error(self, _load, _search):
        result = tr.refresh_topic_bank()
        assert result["added"] == 0
        assert result["error"] is not None


# ── refresh_behavioural_bank ─────────────────────────────────────────────────


class TestRefreshBehaviouralBank:
    @patch.object(tr, "save_behavioural")
    @patch.object(tr, "extract_behavioural_from_search")
    @patch.object(tr, "search_for_behavioural_topics")
    @patch.object(tr, "load_existing_behavioural")
    def test_drops_duplicates(self, mock_load, mock_search, mock_extract, mock_save):
        # Bug B regression: the duplicate question must NOT be re-added.
        dup_q = "Tell me about a time you solved a hard problem."
        mock_load.return_value = [
            {"id": "beh_001", "competency": "problem-solving", "question": dup_q}
        ]
        mock_search.return_value = "raw text"
        mock_extract.return_value = [
            {"competency": "problem-solving", "question": dup_q},  # duplicate
            {"competency": "leadership", "question": "Describe a time you led a team."},
        ]
        result = tr.refresh_behavioural_bank()
        assert result["added"] == 1
        # Saved bank = 1 existing + 1 genuinely new = 2 (not 3).
        saved_list = mock_save.call_args[0][0]
        assert len(saved_list) == 2

    @patch.object(tr, "save_behavioural")
    @patch.object(tr, "extract_behavioural_from_search")
    @patch.object(tr, "search_for_behavioural_topics")
    @patch.object(tr, "load_existing_behavioural")
    def test_groups_new_by_competency(self, mock_load, mock_search, mock_extract, mock_save):
        mock_load.return_value = []
        mock_search.return_value = "raw text"
        mock_extract.return_value = [
            {"competency": "leadership", "question": "Describe a time you led a team."},
            {"competency": "leadership", "question": "Give an example of leading change."},
        ]
        result = tr.refresh_behavioural_bank()
        assert result["added"] == 2
        assert len(result["by_competency"]["leadership"]) == 2
