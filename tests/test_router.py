"""
test_router.py — Unit tests for the conditional edge functions in graph/router.py.

These are pure functions (no LLM, no I/O) that decide which branch the graph
takes after each evaluation node. They are the contract the whole graph relies
on, so the defaults matter as much as the happy path.
"""

from graph.router import (
    route_after_evaluation,
    route_after_rewrite,
    route_after_understanding,
)


class TestRouteAfterEvaluation:
    def test_strong_goes_reinforce(self):
        assert route_after_evaluation({"verdict": "strong"}) == "reinforce"

    def test_weak_goes_explain(self):
        assert route_after_evaluation({"verdict": "weak"}) == "explain"

    def test_unknown_verdict_goes_explain(self):
        assert route_after_evaluation({"verdict": "banana"}) == "explain"

    def test_missing_verdict_defaults_to_explain(self):
        # .get("verdict", "weak") → weak → explain
        assert route_after_evaluation({}) == "explain"


class TestRouteAfterUnderstanding:
    def test_understood_goes_reinforce(self):
        assert route_after_understanding({"verdict": "understand"}) == "reinforce"

    def test_needs_review_goes_review(self):
        assert route_after_understanding({"verdict": "needs_review"}) == "review"

    def test_unknown_verdict_goes_review(self):
        assert route_after_understanding({"verdict": "maybe"}) == "review"

    def test_missing_verdict_defaults_to_review(self):
        assert route_after_understanding({}) == "review"


class TestRouteAfterRewrite:
    def test_retry_requested_loops_back(self):
        assert route_after_rewrite({"retry_requested": True}) == "retry"

    def test_no_retry_updates_profile(self):
        assert route_after_rewrite({"retry_requested": False}) == "update_profile"

    def test_missing_flag_updates_profile(self):
        assert route_after_rewrite({}) == "update_profile"
