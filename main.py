import sqlite3

from langgraph.graph import END, StateGraph

from data.topic_refresher import refresh_behavioural_bank, refresh_topic_bank
from graph.nodes.classifier import classify_intent
from graph.nodes.learn import (
    comprehension_check_node,
    evaluate_understanding_node,
    learning_reinforce_node,
    learning_review_node,
    select_learn_topic_node,
    teach_node,
)
from graph.nodes.revise import (
    ask_question_node,
    evaluate_answer_node,
    explain_node,
    reinforce_node,
    select_topic_node,
)
from graph.nodes.star import (
    critique_node,
    present_question_node,
    rewrite_node,
    star_analyser_node,
    update_star_profile_node,
)
from graph.router import (
    route_after_evaluation,
    route_after_rewrite,
    route_after_understanding,
)
from graph.state import AgentState
from persistence.checkpointer import get_checkpointer


# --- Shared initial state ---
def fresh_state() -> AgentState:
    """Returns a clean initial state with all fields set."""
    return {
        "messages": [],
        "topic_scores": {},
        "topics_weak": [],
        "topics_introduced": [],
        "current_topic": "",
        "current_question": "",
        "user_answer": "",
        "evaluation": "",
        "verdict": "",
        "current_competency": "",
        "star_analysis": {},
        "star_scores": [],
        "competencies_done": [],
        "retry_requested": False,
    }


def show_progress():
    """
    Reads the persisted state and prints a summary of the learner profile.
    Uses both thread IDs to aggregate scores across all modes.
    """
    print("\nYour Progress Summary")
    print("-" * 50)

    checkpointer = get_checkpointer()

    # Pulling states from all 3 thread IDs
    thread_ids = ["learn_main", "revision_main", "star_main"]
    all_states = {}

    for tid in thread_ids:
        config = {"configurable": {"thread_id": tid}}

        try:
            state = checkpointer.get(config)

            if state and state["channel_values"]:
                all_states[tid] = state["channel_values"]
        except Exception:
            pass

    # --- CS / Networking scores ---
    topic_scores = {}
    topics_introduced = []

    for tid in ["learn_main", "revision_main"]:
        st = all_states.get(tid, {})

        topic_scores.update(st.get("topic_scores", {}))

        for tpc in st.get("topics_introduced", []):
            if tpc not in topics_introduced:
                topics_introduced.append(tpc)

    if topic_scores:
        print("\nCS & Networking Topics:")

        for topic, score in sorted(topic_scores.items(), key=lambda x: x[1]):
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            status = "Weak" if score < 0.5 else "Good"
            print(f"{status} [{bar}] {score: .1f} {topic}")
    else:
        print("\nNo CS/Networking session recorded yet.")

    # --- STAR scores ---
    star_state = all_states.get("star_main", {})
    star_scores = star_state.get("star_scores", [])
    competencies_done = star_state.get("competencies_done", [])

    if star_scores:
        avg = round(sum(s["score"] for s in star_scores) / len(star_scores), 1)
        print(f"\n STAR Coaching - {len(star_scores)} session(s) | Average: {avg}/10")
        print(
            f"Competencies practiced: {', '.join(competencies_done) if competencies_done else 'none yet'}"
        )
        print("\nRecent sessions:")

        for entry in star_scores[-3:]:  # Showing last 3
            bar = "█" * entry["score"] + "░" * (10 - entry["score"])
            print(
                f"[{bar}] {entry['score']}/10  {entry['competency']}  ({entry['date']})"
            )

            if entry.get("gap") and entry["gap"].lower() != "none":
                print(f"Gap: {entry['gap']}")
    else:
        print("\nNo STAR sessions recorded yet.")

    print("\n" + "-" * 50 + "\n")


# --- User input helper ---
def collect_multiline_input() -> str:
    """Collects multi-line input - submit with a blank line"""
    lines = []

    while True:
        line = input()

        if line == "":
            break

        lines.append(line)

    return "\n".join(lines)


# --- Revision Session ---


def build_revision_graph() -> StateGraph:
    """Constructs the revision graph."""
    builder = StateGraph(AgentState)

    # Registering every node
    builder.add_node("select_topic", select_topic_node)
    builder.add_node("ask_question", ask_question_node)
    builder.add_node("evaluate_answer", evaluate_answer_node)
    builder.add_node("reinforce", reinforce_node)
    builder.add_node("explain", explain_node)

    # Making the edges
    builder.set_entry_point("select_topic")
    builder.add_edge("select_topic", "ask_question")
    builder.add_edge("ask_question", "evaluate_answer")

    # Interrupt to wait for user response
    builder.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {
            "reinforce": "reinforce",
            "explain": "explain",
        },
    )

    builder.add_edge("reinforce", END)
    builder.add_edge("explain", END)

    return builder


def run_revision_session():
    checkpointer = get_checkpointer()
    builder = build_revision_graph()

    # Interrupt before evaluate answer:
    # select_topic + ask_question, then PAUSES
    # won't proceed to evaluate_answer until invoke() called again
    app = builder.compile(
        checkpointer=checkpointer, interrupt_before=["evaluate_answer"]
    )

    # Each thread_id is an independent session with its own persisted state.
    # using "main" means every run shares the same profile - scores accumulate.
    config = {"configurable": {"thread_id": "revision_main"}}

    print("\nRevision Mode - CS & Networking\n")

    # First invoke: runs until interrupt
    app.invoke(fresh_state(), config=config)

    print("Your answer (press Enter twice to submit):")
    user_answer = collect_multiline_input()

    # Second invoke: resume from checkpoint with the answer
    app.update_state(config, {"user_answer": user_answer})
    app.invoke(None, config=config)  # None - resume, don't re-run from state

    print("\nSession complete.\n")


# --- Learning session ---
def build_learning_graph() -> StateGraph:
    """Constructs the learning graph."""
    builder = StateGraph(AgentState)

    builder.add_node("select_topic", select_learn_topic_node)
    builder.add_node("teach", teach_node)
    builder.add_node("check", comprehension_check_node)
    builder.add_node("evaluate", evaluate_understanding_node)
    builder.add_node("reinforce", learning_reinforce_node)
    builder.add_node("review", learning_review_node)

    builder.set_entry_point("select_topic")
    builder.add_edge("select_topic", "teach")
    builder.add_edge("teach", "check")
    builder.add_edge("check", "evaluate")  # Interrupt here
    builder.add_conditional_edges(
        "evaluate",
        route_after_understanding,
        {
            "reinforce": "reinforce",
            "review": "review",
        },
    )
    builder.add_edge("reinforce", END)
    builder.add_edge("review", END)

    return builder


def run_learning_session():
    checkpointer = get_checkpointer()

    builder = build_learning_graph()
    app = builder.compile(checkpointer=checkpointer, interrupt_before=["evaluate"])

    config = {"configurable": {"thread_id": "learn_main"}}

    print("\nLearning Mode - CS & Networking\n")
    app.invoke(fresh_state(), config=config)

    print("Your explanation (press Enter twice to submit):")
    user_answer = collect_multiline_input()

    app.update_state(config, {"user_answer": user_answer})
    app.invoke(None, config=config)

    print("\nSession complete.\n")


# --- STAR session ---
def build_star_graph() -> StateGraph:
    """Constructs the STAR graph."""
    builder = StateGraph(AgentState)

    builder.add_node("present_question", present_question_node)
    builder.add_node("star_analyser", star_analyser_node)
    builder.add_node("critique", critique_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("update_profile", update_star_profile_node)

    builder.set_entry_point("present_question")
    builder.add_edge("present_question", "star_analyser")  # Interrupt here
    builder.add_edge("star_analyser", "critique")
    builder.add_edge("critique", "rewrite")

    # After rewrite: retry loops to present question, otherwise saves and ends
    builder.add_conditional_edges(
        "rewrite",
        route_after_rewrite,
        {"retry": "present_question", "update_profile": "update_profile"},
    )

    # Interrupt for present_question when looped back for retry
    builder.add_edge("update_profile", END)

    return builder


def run_star_session():
    checkpointer = get_checkpointer()

    builder = build_star_graph()
    app = builder.compile(checkpointer=checkpointer, interrupt_before=["star_analyser"])

    config = {"configurable": {"thread_id": "star_main"}}

    print("\nSTAR Coaching Mode - Behavioural Interview Prep\n")

    # Outer loop handles retry - each retry is a gresh graph invocation
    while True:
        app.invoke(fresh_state(), config=config)

        print("Your answer (press Enter twice to submit):")
        user_answer = collect_multiline_input()

        app.update_state(config, {"user_answer": user_answer})
        app.invoke(None, config=config)

        # Check if user asked to retry - if not, break
        current = app.get_state(config)

        if not current.values.get("retry_requested", False):
            break

        print("\nLet's try again with the feedback in mind...\n")

    print("\n Session complete.\n")


# --- Refresh Session ---


def run_refresh_session():
    """
    Searches the web for current UK interview topics and updates cs_concepts.json.
    Gives a clear summary of what was added.
    """
    print("\nTopic Bank Refresh")
    print("─" * 50)
    print("This will search for current UK tech interview topics")
    print("and add any new ones to your question bank.")
    print("(Uses Tavily search + Gemini — takes about 15-20 seconds)\n")

    confirm = input("Proceed? (y / any other key to cancel): ").strip().lower()

    if confirm != "y":
        print("Cancelled.\n")
        return

    print()
    result = refresh_topic_bank()

    if result.get("error"):
        print(f"\n❌ Refresh failed: {result['error']}\n")
        return

    print("\nTopic bank updated!")
    print(f"Added:  {result['added']} new topic(s)")
    print(f"Total:  {result['total']} topics in bank")

    if result["new_topics"]:
        print("\nNew topics added:")

        for topic in result["new_topics"]:
            print(f" + {topic}")

    print()


# --- Refresh STAR session ---


def run_refresh_star_session():
    """
    Searches the web for UK behavioural interview questions and adds new ones
    to behavioural_questions.json.
    """
    print("\nBehavioural Question Bank Refresh")
    print("─" * 50)
    print("This will search for current UK behavioural interview")
    print("questions and add new ones to your STAR question bank.")
    print("(Uses Tavily search + Gemini — takes about 15-20 seconds)\n")

    confirm = input("Proceed? (y / any other key to cancel): ").strip().lower()

    if confirm != "y":
        print("Cancelled.\n")
        return

    print()
    result = refresh_behavioural_bank()

    if result.get("error"):
        print(f"\n❌ Refresh failed: {result['error']}\n")
        return

    print("\nBehavioural question bank updated!")
    print(f"Added:  {result['added']} new question(s)")
    print(f"Total:  {result['total']} questions in bank")

    if result.get("by_competency"):
        print("\nNew questions by competency:")

        for competency, questions in result["by_competency"].items():
            print(f"\n[{competency}]")

            for q in questions:
                # Truncate long questions for display
                display = q if len(q) <= 80 else q[:77] + "..."
                print(f" + {display}")

    print()


# --- Entry point ---

if __name__ == "__main__":
    print("\nInterview Prep Agent")
    print("─" * 50)
    print("Just tell me what you want to do. Examples:")
    print('  "quiz me on networking"')
    print('  "teach me something new"')
    print('  "I want to practise a behavioural question"')
    print('  "show my progress"')
    print('  "refresh the concept topic bank"')
    print('  "refresh behavioural questions"')
    print('  "exit" to quit')
    print("─" * 50)

    while True:
        print()
        user_input = input(">  ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q", "bye"}:
            print("\nGood luck with the interviews!\n")
            break

        # Classify intent — single LLM call, returns one of four strings
        intent = classify_intent(user_input)
        print(f"\n[Routing to: {intent}]\n")

        if intent == "learn":
            run_learning_session()
        elif intent == "revise":
            run_revision_session()
        elif intent == "star":
            run_star_session()
        elif intent == "progress":
            show_progress()
        elif intent == "refresh":
            run_refresh_session()
        elif intent == "refresh_star":
            run_refresh_star_session()

        # After each session, loop back to the prompt
        print("─" * 50)
        print("What would you like to do next? (or 'exit' to quit)")
