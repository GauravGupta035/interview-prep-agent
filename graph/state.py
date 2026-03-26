import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    # Conversation messages - accumulates across nodes (never overwrites)
    messages: Annotated[list, operator.add]  # fix: = → : so Annotated reducer is applied

    # Current session details
    current_topic: str  # Current topic being covered eg "TCP vs UDP"
    current_question: str  # the question being asked
    user_answer: str  # what the user has typed in response
    evaluation: str  # feedback from the LLM
    verdict: str  # final chosen verdict after the answer's evaluation

    # Learner profile - CS/Networking
    topic_scores: dict  # How much each topic is scored
    topics_weak: list  # topics with score < 0.5
    topics_introduced: list  # Topics which are taught at least once

    # Behavioural profile
    current_competency: str  # eg. "problem-solving"
    star_analysis: dict  # parsed JSON from STAR analyser
    star_scores: list  # [{competency, date, score, gaps}]
    competencies_done: list  # competencies practiced at least once
    retry_requested: bool  # did user ask to try again?
