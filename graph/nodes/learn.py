import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from graph.state import AgentState

# Initialising LLM at module level
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY")
)


def load_questions() -> list:
    path = Path(__file__).parent.parent.parent / "data" / "cs_concepts.json"

    with open(path) as f:
        return json.load(f)


def select_learn_topic_node(state: AgentState) -> dict:
    """
    Picks a topic the user hasn't been taught yet.
    Falls back to weakest-scored topics if everything has been introduced.
    """
    questions = load_questions()
    introduced = state.get("topics_introduced", [])
    topic_scored = state.get("topic_scores", {})

    # Priority 1: topics never introduced
    new_topics = [q for q in questions if q["topic"] not in introduced]

    # Priority 2: If everything introduced, pick the weakest
    if new_topics:
        chosen = random.choice(new_topics)
    else:
        chosen = min(questions, key=lambda q: topic_scored.get(q["topic"], 0))

    print(f"\nLearning {chosen['topic']}")
    print(f"Category: {chosen['category']} | Difficulty: {chosen['difficulty']}\n")

    return {
        "current_topic": chosen["topic"],
        # Storing key points in messages for downstream nodes to reference
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Topic: {chosen['topic']}\n"
                    f"Key points to cover: {', '.join(chosen['key_points'])}\n"
                    f"Common mistakes: {', '.join(chosen['common_mistakes'])}"
                ),
            }
        ],
    }


def teach_node(state: AgentState) -> dict:
    """
    LLM teaches the topic with a clear structure and a real-world analogy.
    This is the 'lecture' - no question asked yet.
    """
    topic = state["current_topic"]

    # Pulling the key points context stored in select_learn_topic_node
    context = ""

    for msg in state.get("message", []):
        if isinstance(msg, dict) and msg.get("role") == "system":
            context = msg["content"]
            break

    prompt = f"""You are an expert CS tutor preparing a candidate for UK tech interviews.

        {context}

        Teach the topic "{topic}" clearly and concisely. Structure your response as:

        CONCEPT:
        [2-3 sentences explaining what it is and why it matters]

        HOW IT WORKS:
        [The key mechanism or process, step by step if appropriate]

        ANALOGY:
        [One real-world analogy that makes it click]

        INTERVIEW ANGLE:
        [One sentence on how this typically comes up in UK tech interviews]

        Keep the total response under 200 words. Be direct — no fluff."""

    response = llm.invoke([HumanMessage(content=prompt)])

    print("-" * 50)
    print(response.content)
    print("-" * 50)

    return {"messages": [{"role": "assistant", "content": response.content}]}


def comprehension_check_node(state: AgentState) -> dict:
    """
    Prompts the user to explain the topic back in their own words.
    The graoh interrupts AFTER this node - waiting for the user's response.
    """
    topic = state["current_topic"]

    print(f"\nNow explain '{topic}' back to me in 2-3 sentences.")
    print("Pretend you're explaining it to a colleague who hasn't heard of it.\n")
    print("(Type your explanation below and press Enter twice when done)\n")

    return {}


def evaluate_understanding_node(state: AgentState) -> dict:
    """
    Evaluates the user's explanation - did they grasp the core concept?
    Scores understanding and identifies specific gaps.
    """
    topic = state["current_topic"]
    explanation = state["user_answer"]

    # Getting the teaching context so it can be used to evaluate against key points
    context = ""

    for msg in state.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "system":
            context = msg["content"]
            break

    prompt = f"""You are evaluating whether a student understood a concept they were just taught.

    {context}

    Their explanation of "{topic}":
    {explanation}

    Evaluate purely on comprehension — did they grasp the core idea from what they were just taught?

    Respond in this exact format:
    SCORE: [0.0-1.0]
    VERDICT: [understood/needs_review]
    FEEDBACK: [2 sentences — what they got right, what was missing or imprecise]
    GAP: [the single most important thing they missed, or "none"]
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content

    # Parse score and verdict
    score = 0.5  # Default fallback
    verdict = "needs_review"

    for line in result.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except ValueError:
                pass

        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT", "").strip().lower()

    # Update profile
    topic_scores = state.get("topic_scores", {})
    topics_introduced = state.get("topics_introduced", [])

    existing = topic_scores.get(topic, score)
    topic_scores[topic] = round((existing + score) / 2, 2)

    if topic not in topics_introduced:
        topics_introduced += [topic]

    return {
        "evaluation": result,
        "verdict": verdict,
        "topic_scores": topic_scores,
        "topic_introduced": topics_introduced,
        "messages": [{"role": "assistant", "content": result}],
    }


def learning_reinforce_node(state: AgentState) -> dict:
    """User understood - affirm and give one thing to deepen knowledge."""
    topic = state["current_topic"]
    score = state.get("topic_scores", {}).get(topic, 0)

    print(f"\nGood grasp of {topic}! (Score: {score})\n")
    print(state["evaluation"])
    print(
        "\nNext step: try the revision mode to get scenario-based questions on this topic.\n"
    )

    return {}


def learning_review_node(state: AgentState) -> dict:
    """User needs another pass - show gaps and re-explain the key point they missed."""
    topic = state["current_topic"]

    print(f"\nNot quite - let's fill the gap on {topic}:\n")
    print(state["evaluation"])

    # Extract the GAP line and give a targeted re-explanation
    gap = ""

    for line in state["evaluation"].split("\n"):
        if line.startswith("GAP:"):
            gap = line.replace("GAP:", "").strip()
            break

    if gap and gap.lower() != "none":
        prompt = f"""The student just learned about "{topic}" but missed this key point: {gap}

        Give a single, focused 3-sentence explanation of just this gap.
        Use a concrete example. Be direct."""

    print("Run 'learn' mode again to try this topic once more.\n")

    return {}
