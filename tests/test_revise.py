"""
test_revise.py — Unit tests for the revision-path nodes.

Covers topic selection (weak-first prioritisation), the SCORE/VERDICT parsing,
the moving-average scoring, and the key-points context plumbing.

The context test is the regression guard for Bug A: select_topic_node used to
store a SystemMessage object that evaluate_answer_node could never read back,
so every answer was graded with no key points.
"""

from unittest.mock import MagicMock, patch

from graph.nodes import revise


def _question(topic, qtypes=None):
    return {
        "topic": topic,
        "category": "networking",
        "difficulty": "medium",
        "question_types": qtypes or [{"type": "conceptual", "q": f"What is {topic}?"}],
        "key_points": ["kp1", "kp2"],
        "common_mistakes": ["cm1"],
    }


# ── select_topic_node ───────────────────────────────────────────────────────


class TestSelectTopicNode:
    @patch.object(revise.random, "choice", side_effect=lambda pool: pool[0])
    @patch.object(revise, "load_questions")
    def test_weak_topics_prioritised(self, mock_load, _choice):
        mock_load.return_value = [_question("A"), _question("B")]
        # A is strong (0.9), B is weak (0.2) → only B is in the weak pool,
        # so choice([B]) must return B.
        out = revise.select_topic_node({"topic_scores": {"A": 0.9, "B": 0.2}})
        assert out["current_topic"] == "B"

    @patch.object(revise.random, "choice", side_effect=lambda pool: pool[0])
    @patch.object(revise, "load_questions")
    def test_falls_back_to_full_pool_when_none_weak(self, mock_load, _choice):
        mock_load.return_value = [_question("A"), _question("B")]
        out = revise.select_topic_node({"topic_scores": {"A": 0.9, "B": 0.9}})
        # No weak topics → pool is all questions → choice picks the first.
        assert out["current_topic"] == "A"

    @patch.object(revise.random, "choice", side_effect=lambda pool: pool[0])
    @patch.object(revise, "load_questions")
    def test_stores_context_as_dict_not_object(self, mock_load, _choice):
        # Bug A regression: the system message must be a dict so the
        # evaluator's `isinstance(msg, dict)` check can find it.
        mock_load.return_value = [_question("TCP")]
        out = revise.select_topic_node({"topic_scores": {}})
        msg = out["messages"][0]
        assert isinstance(msg, dict)
        assert msg["role"] == "system"
        assert "kp1" in msg["content"]


# ── evaluate_answer_node ─────────────────────────────────────────────────────


class TestEvaluateAnswerNode:
    @patch.object(revise, "llm")
    def test_parses_score_and_verdict(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="SCORE: 0.8\nVERDICT: strong\nFEEDBACK: good\nMISSING: none"
        )
        out = revise.evaluate_answer_node(
            {
                "current_topic": "TCP",
                "current_question": "q",
                "user_answer": "a",
                "topic_scores": {},
                "messages": [],
            }
        )
        assert out["verdict"] == "strong"
        # First attempt: existing defaults to score → (0.8 + 0.8) / 2 = 0.8
        assert out["topic_scores"]["TCP"] == 0.8

    @patch.object(revise, "llm")
    def test_moving_average_blends_with_existing(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="SCORE: 0.8\nVERDICT: strong"
        )
        out = revise.evaluate_answer_node(
            {
                "current_topic": "TCP",
                "current_question": "q",
                "user_answer": "a",
                "topic_scores": {"TCP": 0.4},
                "messages": [],
            }
        )
        assert out["topic_scores"]["TCP"] == 0.6  # (0.4 + 0.8) / 2

    @patch.object(revise, "llm")
    def test_malformed_score_falls_back_to_default(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="SCORE: not-a-number\nVERDICT: weak"
        )
        out = revise.evaluate_answer_node(
            {
                "current_topic": "TCP",
                "current_question": "q",
                "user_answer": "a",
                "topic_scores": {},
                "messages": [],
            }
        )
        # Falls back to the default 0.5 rather than crashing.
        assert out["topic_scores"]["TCP"] == 0.5
        assert out["verdict"] == "weak"

    @patch.object(revise, "llm")
    def test_key_points_context_reaches_the_prompt(self, mock_llm):
        # Bug A regression: a dict system message must be picked up and
        # injected into the evaluation prompt.
        mock_llm.invoke.return_value = MagicMock(content="SCORE: 0.5\nVERDICT: weak")
        revise.evaluate_answer_node(
            {
                "current_topic": "TCP",
                "current_question": "q",
                "user_answer": "a",
                "topic_scores": {},
                "messages": [
                    {"role": "system", "content": "Key points for TCP: handshake"}
                ],
            }
        )
        sent_prompt = mock_llm.invoke.call_args[0][0][0].content
        assert "Key points for TCP: handshake" in sent_prompt
