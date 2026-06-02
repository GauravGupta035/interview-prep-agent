"""
test_classifier.py — Unit tests for the intent classifier.

The LLM is mocked, so these assert routing behaviour and the safety net:
anything the model returns that isn't a known intent must fall back to "revise".
"""

from unittest.mock import MagicMock, patch

import pytest

from graph.nodes import classifier


def _mock_response(text):
    return MagicMock(content=text)


@pytest.mark.parametrize(
    "llm_output,expected",
    [
        ("learn", "learn"),
        ("revise", "revise"),
        ("star", "star"),
        ("progress", "progress"),
        ("refresh", "refresh"),
        ("refresh_star", "refresh_star"),
    ],
)
@patch.object(classifier, "llm")
def test_each_valid_intent_passes_through(mock_llm, llm_output, expected):
    mock_llm.invoke.return_value = _mock_response(llm_output)
    assert classifier.classify_intent("whatever") == expected


@patch.object(classifier, "llm")
def test_strips_whitespace_and_lowercases(mock_llm):
    mock_llm.invoke.return_value = _mock_response("  LEARN\n")
    assert classifier.classify_intent("teach me") == "learn"


@patch.object(classifier, "llm")
def test_unknown_intent_falls_back_to_revise(mock_llm):
    mock_llm.invoke.return_value = _mock_response("banana")
    assert classifier.classify_intent("???") == "revise"


@patch.object(classifier, "llm")
def test_empty_response_falls_back_to_revise(mock_llm):
    mock_llm.invoke.return_value = _mock_response("")
    assert classifier.classify_intent("???") == "revise"


@patch.object(classifier, "llm")
def test_extra_punctuation_is_not_valid_and_falls_back(mock_llm):
    # The prompt asks for a bare word; "learn." should NOT match and fall back.
    mock_llm.invoke.return_value = _mock_response("learn.")
    assert classifier.classify_intent("teach me") == "revise"
