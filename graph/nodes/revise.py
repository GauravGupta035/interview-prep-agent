import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from graph.state import AgentState

# Initialising LLM at module level
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY")
)


def load_questions() -> list:
    """Load the question bank from JSON"""
    path = Path(__file__).parent.parent.parent / "data" / "cs_concepts.json"
    with open(path) as f:
        return json.load(f)


def select_topic_node(state: AgentState) -> dict:
    """
    Picks a topic to revise.
    Priority: weak topics first, then random from the rest.
    """
    questions = load_questions()
    topic_scores = state.get("topic_scores", {})

    # Separating weak topics from the rest
    weak = [q for q in questions if topic_scores.get(q["topic"], 0) < 0.5]
    pool = weak if weak else questions

    chosen = random.choice(pool)

    # Picking a random question type from the entry
    question_entry = random.choice(chosen["question_types"])

    print(f"\n Topic: {chosen['topic']}")
    print(f"Category: {chosen['category']} | Difficulty: {chosen['difficulty']}\n")

    return {
        "current_topic": chosen["topic"],
        "current_question": question_entry["q"],
        # Storing key points so evaluators can reference them
        "messages": [
            SystemMessage(
                f"Key points for {chosen['topic']}: {', '.join(chosen['key_points'])}. Common mistakes: {', '.join(chosen['common_mistakes'])}"
            )
        ],
    }


def ask_question_node(state: AgentState) -> dict:
    """
    Presents the question to the user.
    The graph will interrupt AFTER this node, waiting for user input.
    """
    question = state["current_question"]

    print("-" * 50)
    print(f"Q: {question}")
    print("-" * 50)
    print("(Type your answer below and press Enter twice when done)\n")

    # Nothing to return yet, user answer comes via the interrupt
    return {}


def evaluate_answer_node(state: AgentState) -> dict:
    """
    Evaluates the user's answer against the key points.
    Returns structured feedback and a score.
    """
    topic = state["current_topic"]
    question = state["current_question"]
    answer = state["user_answer"]

    # Find key points for this topic from messages
    context = ""
    for msg in state.get("message", []):
        if isinstance(msg, dict) and msg.get("role") == "system":
            context = msg["content"]
            break

    prompt = f"""You are an expert UK technical interviewer evaluating a candidate's answer.
        Topic: {topic}
        Question asked: {question}
        {context}

        Candidate's answer:
        {answer}

        Evaluate the answer on:
        1. Technical accuracy
        2. Depth of understanding
        3. Whether key points were covered
        4. Any misconceptions

        Respond in this exact format:
        SCORE: [0.0-1.0]
        VERDICT: [strong/weak]
        FEEDBACK: [2-3 sentences of specific, constructive feedback]
        MISSING: [key points that were missing, or "none"]
        """

    response = llm.invoke([HumanMessage(content=prompt)])
    evaluation_text = response.content

    # Parsing the score from the response
    score = 0.5  # default fallback
    verdict = "weak"

    for line in evaluation_text.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except ValueError:
                pass

        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip().lower()

    # Updating the topic score (simple moving average)
    topic_scores = state.get("topic_scores", {})
    existing = topic_scores.get(topic, score)
    topic_scores[topic] = round((existing + score) / 2, 2)

    return {
        "evaluation": evaluation_text,
        "verdict": verdict,
        "topic_scores": topic_scores,
        "messages": [{"role": "assistant", "content": evaluation_text}],
    }


def reinforce_node(state: AgentState) -> dict:
    """Called when the answer is strong. It affirms and adds depth."""
    print("\n Strong answer!\n")
    print(state["evaluation"])
    print(
        "\n Want to go deeper? Try explain this in a system design context next time.\n"
    )
    return {}


def explain_node(state: AgentState) -> dict:
    """Called when the answer was weak. Explains gaps clearly"""
    print("\n Needs work - here's the breakdown:\n")
    print(state["evaluation"])
    print("\n Revisit the topic and try again in your next session.\n")
    return {}
