"""
test_graphs.py — Graph wiring smoke tests + one end-to-end golden path.

The compile tests catch broken edges / typo'd node names without any LLM.
The end-to-end test drives the full revision graph (select → ask → interrupt
→ evaluate → branch → END) with a mocked LLM and an in-memory checkpointer,
asserting that a complete turn updates the persisted profile.
"""

from unittest.mock import MagicMock, patch

from langgraph.checkpoint.memory import MemorySaver

from graph.nodes import revise
from main import (
    build_learning_graph,
    build_revision_graph,
    build_star_graph,
    fresh_state,
)


# ── Compile / wiring smoke tests ─────────────────────────────────────────────


class TestGraphsCompile:
    def test_revision_graph_has_expected_nodes(self):
        app = build_revision_graph().compile(
            checkpointer=MemorySaver(), interrupt_before=["evaluate_answer"]
        )
        nodes = app.get_graph().nodes
        for name in ["select_topic", "ask_question", "evaluate_answer", "reinforce", "explain"]:
            assert name in nodes

    def test_learning_graph_compiles(self):
        app = build_learning_graph().compile(
            checkpointer=MemorySaver(), interrupt_before=["evaluate"]
        )
        nodes = app.get_graph().nodes
        for name in ["select_topic", "teach", "check", "evaluate", "reinforce", "review"]:
            assert name in nodes

    def test_star_graph_compiles(self):
        app = build_star_graph().compile(
            checkpointer=MemorySaver(), interrupt_before=["star_analyser"]
        )
        nodes = app.get_graph().nodes
        for name in ["present_question", "star_analyser", "critique", "rewrite", "update_profile"]:
            assert name in nodes


# ── End-to-end golden path (revision) ────────────────────────────────────────


class TestRevisionGoldenPath:
    @patch.object(revise.random, "choice", side_effect=lambda pool: pool[0])
    @patch.object(revise, "load_questions")
    @patch.object(revise, "llm")
    def test_full_turn_updates_profile(self, mock_llm, mock_load, _choice):
        mock_load.return_value = [
            {
                "topic": "TCP vs UDP",
                "category": "networking",
                "difficulty": "medium",
                "question_types": [{"type": "conceptual", "q": "What is TCP?"}],
                "key_points": ["reliability", "handshake"],
                "common_mistakes": ["cm1"],
            }
        ]
        mock_llm.invoke.return_value = MagicMock(
            content="SCORE: 0.8\nVERDICT: strong\nFEEDBACK: solid\nMISSING: none"
        )

        app = build_revision_graph().compile(
            checkpointer=MemorySaver(), interrupt_before=["evaluate_answer"]
        )
        config = {"configurable": {"thread_id": "t1"}}

        # First invoke runs up to the interrupt (before evaluate_answer).
        app.invoke(fresh_state(), config=config)

        # Supply the user's answer and resume.
        app.update_state(config, {"user_answer": "TCP is reliable, UDP is not."})
        app.invoke(None, config=config)

        final = app.get_state(config).values
        assert final["current_topic"] == "TCP vs UDP"
        assert final["verdict"] == "strong"
        assert final["topic_scores"]["TCP vs UDP"] == 0.8
