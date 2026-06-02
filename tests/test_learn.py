"""
test_learn.py — Unit tests for the learning-path nodes.

Covers new-topic-first selection, the weakest-topic fallback, and the
SCORE/VERDICT parsing + profile update in evaluate_understanding_node.
"""

from unittest.mock import MagicMock, patch

from graph.nodes import learn


def _question(topic):
    return {
        "topic": topic,
        "category": "systems",
        "difficulty": "medium",
        "key_points": ["kp1", "kp2"],
        "common_mistakes": ["cm1"],
    }


# ── select_learn_topic_node ──────────────────────────────────────────────────


class TestSelectLearnTopicNode:
    @patch.object(learn.random, "choice", side_effect=lambda pool: pool[0])
    @patch.object(learn, "load_questions")
    def test_picks_a_never_introduced_topic_first(self, mock_load, _choice):
        mock_load.return_value = [_question("A"), _question("B")]
        out = learn.select_learn_topic_node({"topics_introduced": ["A"]})
        # A already introduced → only B is "new" → choice([B]) returns B.
        assert out["current_topic"] == "B"

    @patch.object(learn, "load_questions")
    def test_falls_back_to_weakest_when_all_introduced(self, mock_load):
        mock_load.return_value = [_question("A"), _question("B")]
        out = learn.select_learn_topic_node(
            {
                "topics_introduced": ["A", "B"],
                "topic_scores": {"A": 0.9, "B": 0.1},
            }
        )
        # Everything introduced → min() by score → B.
        assert out["current_topic"] == "B"

    @patch.object(learn.random, "choice", side_effect=lambda pool: pool[0])
    @patch.object(learn, "load_questions")
    def test_stores_context_dict(self, mock_load, _choice):
        mock_load.return_value = [_question("DNS")]
        out = learn.select_learn_topic_node({})
        msg = out["messages"][0]
        assert msg["role"] == "system"
        assert "DNS" in msg["content"]


# ── evaluate_understanding_node ──────────────────────────────────────────────


class TestEvaluateUnderstandingNode:
    @patch.object(learn, "llm")
    def test_parses_verdict_and_marks_introduced(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="SCORE: 0.9\nVERDICT: understood\nGAP: none"
        )
        out = learn.evaluate_understanding_node(
            {
                "current_topic": "DNS",
                "user_answer": "explanation",
                "topic_scores": {},
                "topics_introduced": [],
                "messages": [],
            }
        )
        assert out["verdict"] == "understood"
        assert "DNS" in out["topics_introduced"]
        assert out["topic_scores"]["DNS"] == 0.9

    @patch.object(learn, "llm")
    def test_does_not_duplicate_introduced_topic(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="SCORE: 0.5\nVERDICT: needs_review"
        )
        out = learn.evaluate_understanding_node(
            {
                "current_topic": "DNS",
                "user_answer": "x",
                "topic_scores": {"DNS": 0.5},
                "topics_introduced": ["DNS"],
                "messages": [],
            }
        )
        assert out["topics_introduced"].count("DNS") == 1

    @patch.object(learn, "llm")
    def test_malformed_score_falls_back(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="SCORE: ???\nVERDICT: needs_review"
        )
        out = learn.evaluate_understanding_node(
            {
                "current_topic": "DNS",
                "user_answer": "x",
                "topic_scores": {},
                "topics_introduced": [],
                "messages": [],
            }
        )
        assert out["topic_scores"]["DNS"] == 0.5
