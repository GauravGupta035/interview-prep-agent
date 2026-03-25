from graph.state import AgentState


def route_after_evaluation(state: AgentState) -> str:
    """
    Revision path: strong answer -> reinforce, weak -> explain,
    """
    verdict = state.get("verdict", "weak")

    if verdict == "strong":
        return "reinforce"

    return "explain"


def route_after_understanding(state: AgentState) -> str:
    """
    Learning path: understand -> reinforce, needs_review -> review.
    """
    verdict = state.get("verdict", "needs_review")

    if verdict == "understand":
        return "reinforce"

    return "review"


def route_after_rewrite(state: AgentState) -> str:
    """
    STAR path: if user wants to retry -> back to present_question,
    otherwise -> update profile and end.
    """
    return "retry" if state.get("retry_requested", False) else "update_profile"
