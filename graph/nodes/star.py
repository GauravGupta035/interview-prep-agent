import json
import os
import random
from datetime import date
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


def load_behavioural_questions() -> list:
    path = Path(__file__).parent.parent.parent / "data" / "behavioural_questions.json"
    with open(path) as f:
        return json.load(f)


def present_question_node(state: AgentState) -> dict:
    """
    Picks a competency-based question.
    Prioritises competencies not yet practiced.
    """
    questions = load_behavioural_questions()
    competencies_done = state.get("competencies_done", [])

    # Priority: competencies not yet practiced
    fresh = [q for q in questions if q["competency"] not in competencies_done]
    pool = fresh if fresh else questions

    chosen = random.choice(pool)

    print("\n" + "─" * 50)
    print(f"Competency: {chosen['competency'].upper()}")
    print("─" * 50)
    print(f"\n{chosen['question']}\n")
    print("Tip: Use the STAR format — Situation, Task, Action, Result")
    print("'I did' not 'we did' — be specific about your personal contribution")
    print("─" * 50)
    print("\n(Type your answer below and press Enter twice when done)\n")

    return {
        "current_question": chosen["question"],
        "current_competency": chosen["competency"],
        # Storing the expectations - used in analysis
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Competency: {chosen['competency']}\n"
                    f"What interviewers want: {', '.join(chosen['what_they_want'])}\n"
                    f"Red flags to avoid: {', '.join(chosen['red_flags'])}"
                ),
            }
        ],
    }


def star_analyser_node(state: AgentState) -> dict:
    """
    The core of the STAR path.
    Uses structured JSON extraction to score each STAR element independently.
    This gives precise, actionable feedback rather than vague commentary.
    """
    question = state["current_question"]
    answer = state["user_answer"]
    competency = state.get("current_competency", "")

    # Get what the interviewers is looking for
    context = ""
    for msg in state.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "system":
            context = msg["content"]
            break

    prompt = f"""You are an expert UK competency interview coach.
    Analyse this answer strictly for STAR structure and content quality.

    Question: {question}
    {context}

    Candidate's answer:
    {answer}

    Analyse each STAR element and return ONLY valid JSON — no preamble, no markdown fences.

    {{
      "situation": {{
        "present": true or false,
        "quality": "strong" or "weak" or "missing",
        "feedback": "specific one-sentence observation"
      }},
      "task": {{
        "present": true or false,
        "quality": "strong" or "weak" or "missing",
        "feedback": "specific one-sentence observation"
      }},
      "action": {{
        "present": true or false,
        "quality": "strong" or "weak" or "missing",
        "feedback": "specific one-sentence observation — flag 'we did' language if present"
      }},
      "result": {{
        "present": true or false,
        "quality": "strong" or "weak" or "missing",
        "feedback": "specific one-sentence observation — flag if unquantified"
      }},
      "overall_score": a number from 1 to 10,
      "biggest_gap": "the single most important thing to fix",
      "enhanced_version": "a full rewritten version of the answer in clean STAR format, 150-200 words, first person, specific actions"
    }}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Strip markdown fences if the LLM adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]

        if raw.startswith("json"):
            raw = raw[4:]

    raw = raw.strip()

    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if JSON is malformed
        analysis = {
            "situation": {
                "present": False,
                "quality": "missing",
                "feedback": "Could not parse",
            },
            "task": {
                "present": False,
                "quality": "missing",
                "feedback": "Could not parse",
            },
            "action": {
                "present": False,
                "quality": "missing",
                "feedback": "Could not parse",
            },
            "result": {
                "present": False,
                "quality": "missing",
                "feedback": "Could not parse",
            },
            "overall_score": 5,
            "biggest_gap": "Could not fully parse your answer — try again",
            "enhanced_version": answer,
        }

    return {
        "star_analysis": analysis,
        "messages": [{"role": "assistant", "content": raw}],
    }


def critique_node(state: AgentState) -> dict:
    """
    Displays the STAR breakdown clearly - element by element.
    Uses Rich-style formatting via plain characters.
    """
    analysis = state.get("star_analysis", {})  # fix: typo "satr_analysis" → "star_analysis"
    score = analysis.get("overall_score", 0)

    # Score bar - visual indicator
    filled = int(score)
    bar = "█" * filled + "░" * (10 - filled)

    print(f"\nSTAR Analysis [{bar}] {score}/10")
    print("-" * 50)

    elements = ["situation", "task", "action", "result"]
    labels = {
        "situation": "S — Situation",
        "task": "T — Task",
        "action": "A — Action",
        "result": "R — Result",
    }

    for el in elements:
        data = analysis.get(el, {})
        quality = data.get("quality", "missing")
        feedback = data.get("feedback", "")

        print(f"\n{labels[el]} [{quality.upper()}]")
        print(f"{feedback}")

    print(f"\nBiggest gap: {analysis.get('biggest_gap', '')}")
    print("-" * 50)

    return {}


def rewrite_node(state: AgentState) -> dict:
    """
    Shows the LLM-enhanced STAR version of the answer.
    Then asks whether the user wants to try again or move on.
    """
    analysis = state.get("star_analysis", {})
    enhanced = analysis.get("enhanced_version", "")
    score = analysis.get("overall_score", 0)

    print("\nEnhanced STAR version:\n")
    print("-" * 50)
    print(enhanced)
    print("-" * 50)

    if score < 7:
        print("\nWould you like to try this question again with the feedback in mind?")
    else:
        print("\nSolid answer! Try another competency or move on.")

    print("(y = try same question again | any other key = move on)\n")

    choice = input("> ").strip().lower()
    retry = choice == "y"

    return {"retry_requested": retry}  # fix: typo "retry)requested" → "retry_requested"


def update_star_profile_node(state: AgentState) -> dict:
    """
    Saves this session's STAR score to the persistent profile.
    Updates competencies_done and star_scores lists.
    """
    analysis = state.get("star_analysis", {})
    competency = state.get("current_competency", "")
    score = analysis.get("overall_score", 0)
    biggest_gap = analysis.get("biggest_gap", "")

    star_scores = list(state.get("star_scores", []))
    competencies_done = list(state.get("competencies_done", []))

    # Append this session's record
    star_scores.append(
        {
            "competency": competency,
            "date": str(date.today()),
            "score": score,
            "gap": biggest_gap,
        }
    )

    # Track competency as practiced
    if competency not in competencies_done:
        competencies_done.append(competency)

    avg = round(sum(s["score"] for s in star_scores) / len(star_scores), 1)

    print(f"\nProfile updated = competencies practiced: {len(competencies_done)}")
    print(f"Average STAR score across all sessions: {avg}/10\n")

    return {"star_scores": star_scores, "competencies_done": competencies_done}
