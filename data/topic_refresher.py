import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

# Initialising LLM at module level
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY")
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

CONCEPTS_PATH = Path(__file__).parent / "cs_concepts.json"
BEHAVIOURAL_PATH = Path(__file__).parent / "behavioural_questions.json"

# Quesries that are targeted at UK tech interviews specifically
SEARCH_QUERIES = [
    "must know networking topics UK software engineer interviews 2026",
    "common system design concepts UK tech interviews graduate",
    "software engineering concepts asked in fintech UK interviews",
    "CS fundamentals questions UK FAANG interviews 2026",
]


def load_existing_concepts() -> list:
    with open(CONCEPTS_PATH) as f:
        return json.load(f)


def save_concepts(concepts: list) -> None:
    with open(CONCEPTS_PATH, "w") as f:
        json.dump(concepts, f, indent=2)


def search_for_topics() -> str:
    """
    Runs multiple targeted Tavily searches and concatenates the results into a
    single text blob for the LLM to reason over.
    """
    print("Searching for current UK interview topics...")
    combined = []

    for query in SEARCH_QUERIES:
        print(f"-> {query}")

        try:
            results = tavily.search(query=query, max_results=4, search_depth="basic")

            for res in results.get("results", []):
                # Only using the content field - url and title aren't needed
                combined.append(res.get("content", ""))
        except Exception as e:
            print(f"Search failed for this query: {e}")
            continue

    return "\n\n".join(combined)


def extract__topics_from_search(raw_text: str, existing_topics: list) -> list:
    """
    Sends the raw search results to the LLM and asks it to extract structed topic
    entries in the exact JSON format.
    We pass in existing topic names so the LLM can avoid duplicates
    """
    existing_names = [t["topic"] for t in existing_topics]

    prompt = f"""You are building a question bank for UK software engineering interview preparation.

    Below is raw text from web searches about must-know CS and networking topics for UK tech interviews (especially fintech/finance firms like BlackRock, JPMorgan, Palantir).

    Your task: extract 8-12 distinct, important topics that are NOT already in the existing list.

    EXISTING TOPICS (do not repeat these):
    {json.dumps(existing_names, indent=2)}

    RAW SEARCH RESULTS:
    {raw_text[:6000]}

    For each new topic, produce a JSON entry in EXACTLY this format:
    {{
      "id": "topic_slug_001",
      "topic": "Topic Name",
      "category": "networking" or "systems" or "databases" or "software_engineering" or "security" or "cloud",
      "difficulty": "easy" or "medium" or "hard",
      "question_types": [
        {{"type": "conceptual", "q": "A conceptual question about this topic?"}},
        {{"type": "scenario",   "q": "A realistic scenario-based question?"}},
        {{"type": "compare",    "q": "A comparison or trade-off question?"}}
      ],
      "key_points": ["point1", "point2", "point3", "point4"],
      "common_mistakes": ["mistake1", "mistake2"]
    }}

    Return ONLY a valid JSON array of these entries.
    No preamble, no markdown fences, no explanation.
    Ensure every topic is genuinely relevant to UK software engineering interviews."""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]

        if raw.startswith("json"):
            raw = raw[4:]

    raw = raw.strip()

    try:
        new_topics = json.loads(raw)

        if not isinstance(new_topics, list):
            raise ValueError("Expected a JSON array")

        return new_topics
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Could not parse LLM response: {e}")
        return []


def deduplicate_concepts(existing: list, new_topics: list) -> list:
    """
    Final de-deuplication pass (safety net) - compares topic names case-insensitively
    """
    existing_names = {t["topic"].lower() for t in existing}
    filtered = []

    for topic in new_topics:
        name = topic.get("topic", "").lower()

        if name and name not in existing_names:
            filtered.append(topic)
            existing_names.add(name)  # Prevents duplicates within the same batch

    return filtered


def assign_concept_ids(existing: list, new_topics: list) -> list:
    """
    Assigns stable IDs to new entries, continuing from the highest existing ID.
    format: category_NNN eg. networking_006
    """
    # Finding the highest numeric suffix per category
    category_counts = {}

    for entry in existing:
        parts = entry.get("id", "").rsplit("_", 1)

        if len(parts) == 2:
            cat = parts[0]
            try:
                n = int(parts[1])
                category_counts[cat] = max(category_counts.get(cat, 0), n)
            except ValueError:
                pass

    for entry in new_topics:
        cat = entry.get("category", "general")
        category_counts[cat] = category_counts.get(cat, 0) + 1
        entry["id"] = f"{cat}_{str(category_counts[cat]).zfill(3)}"

    return new_topics


def refresh_topic_bank() -> dict:
    """
    Main entry point - searches, extracts, duplicates, and saves.
    Returns a summary dict for the caller to display.
    """
    existing = load_existing_concepts()
    count_before = len(existing)

    # Step 1: search
    raw_text = search_for_topics()

    if not raw_text.strip():
        return {
            "added": 0,
            "total": count_before,
            "error": "All searches failed - check Tavily API Key",
        }

    # Step 2: extract structured topics from search results
    print("\nExtracting and structuring new topics...")
    new_topics = extract__topics_from_search(raw_text, existing)

    if not new_topics:
        return {
            "added": 0,
            "total": count_before,
            "error": "LLM extraction returned no topics",
        }

    # Step 3: De-duplicate
    unique_new_topics = deduplicate_concepts(existing, new_topics)

    # Step 4: assign clean IDs
    unique_new_topics = assign_concept_ids(existing, unique_new_topics)

    # Step 5: save
    updated = existing + unique_new_topics
    save_concepts(updated)

    return {
        "added": len(unique_new_topics),
        "total": len(updated),
        "new_topics": [t["topic"] for t in unique_new_topics],
        "error": None,
    }


# --- Behavioural question refresh ---

BEHAVIOURAL_SEARCH_QUERIES = [
    "UK competency based interview questions software engineer 2026",
    "behavioural interview questions fintech UK",
    "STAR format interview questions UK graduate software engineering",
    "common behavioural interview questions UK tech firms 2026",
]

COMPETENCY_CATEGORIES = [
    "problem-solving",
    "learning-agility",
    "stakeholder-management",
    "leadership",
    "resilience",
    "delivery",
    "collaboration",
    "initiative",
    "communication",
    "adaptability",
    "ownership",
    "attention-to-detail",
]


def load_existing_behavioural() -> list:
    with open(BEHAVIOURAL_PATH) as f:
        return json.load(f)


def save_behavioural(questions: list) -> None:
    with open(BEHAVIOURAL_PATH, "w") as f:
        json.dump(questions, f, indent=2)


def search_for_behavioural_topics() -> str:
    """Searches for current UK behavioural interview questions."""
    print("Searching for current UK behavioural interview questions...")
    combined = []

    for query in BEHAVIOURAL_SEARCH_QUERIES:
        print(f"-> {query}")

        try:
            results = tavily.search(query=query, max_results=4, search_depth="basic")

            for res in results.get("results", []):
                combined.append(res.get("content", ""))
        except Exception as e:
            print(f"Search failed for this query: {e}")
            continue

    return "\n\n".join(combined)


def extract_behavioural_from_search(raw_text: str, existing: list) -> list:
    """
    Extracts new behavioural questions from search results.
    Passes existing questions and competencies so the LLM avoids both
    duplication questions and already-covered competencies.
    """
    existing_questions = [q["question"] for q in existing]
    existing_competencies = [q["competency"] for q in existing]

    # Prioritising under-represented competencies
    competency_counts = {}

    for c in existing_competencies:
        competency_counts[c] = competency_counts.get(c, 0) + 1

    underrepresented = [
        c for c in COMPETENCY_CATEGORIES if competency_counts.get(c, 0) < 2
    ]

    prompt = f"""You are building a behavioural interview question bank for UK software engineering roles.

    Focus especially on these underrepresented competencies:
    {json.dumps(underrepresented, indent=2)}

    All competency categories available:
    {json.dumps(COMPETENCY_CATEGORIES, indent=2)}

    EXISTING QUESTIONS (do not repeat or paraphrase these):
    {json.dumps(existing_questions, indent=2)}

    RAW SEARCH RESULTS:
    {raw_text[:6000]}

    Extract 6-10 new, distinct behavioural questions relevant to UK tech interviews.
    Prioritise the underrepresented competencies listed above.

    Return ONLY a valid JSON array in EXACTLY this format — no preamble, no markdown:
    [
      {{
        "id": "beh_NNN",
        "competency": "one of the competency categories listed above",
        "question": "Tell me about a time you...",
        "what_they_want": ["quality1", "quality2", "quality3"],
        "red_flags": ["red_flag1", "red_flag2"]
      }}
    ]

    Rules:
    - Questions must start with "Tell me about a time", "Describe a situation", or "Give an example of"
    - Each competency should map exactly to one of the listed categories
    - what_they_want and red_flags must each have 2-3 items
    - Questions must be realistic for a UK fintech/tech interview"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Stripping markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]

        if raw.startswith("json"):
            raw = raw[4:]

    raw = raw.strip()

    try:
        new_questions = json.loads(raw)
        if not isinstance(new_questions, list):
            raise ValueError("Exoected a JSON array")

        return new_questions
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Could not parse LLM response: {e}")
        return []


def deduplicate_behavioural(existing: list, new_questions: list) -> list:
    """
    De-duplicates by question text - case-insensitive, first 60 chars.
    Shorter comparison avoids false negatives from minor wording differences.
    """
    existing_fingerprints = {q["question"].lower()[:60] for q in existing}

    filtered = []

    for q in new_questions:
        fingerprint = q.get("question", "").lower()[:60]

        if fingerprint and fingerprint not in existing_fingerprints:
            filtered.append(q)
            existing_fingerprints.add(fingerprint)

    return filtered


def assign_behavioural_ids(existing: list, new_questions: list) -> list:
    """Assigns sequential beh_NNN IDs continuing from the highest existing"""
    highest = 0

    for entry in existing:
        id_str = entry.get("id", "")

        if id_str.startswith("beh_"):
            try:
                n = int(id_str.replace("beh_", ""))
                highest = max(highest, n)
            except ValueError:
                pass

    for i, entry in enumerate(new_questions, start=highest + 1):
        entry["id"] = f"beh_{str(i).zfill(3)}"

    return new_questions


def refresh_behavioural_bank() -> dict:
    """
    Main entry point for behavioural refresh.
    Mirrors the structure of refresh_topic_bank().
    """
    existing = load_existing_behavioural()
    count_before = len(existing)

    raw_text = search_for_behavioural_topics()

    if not raw_text.strip():
        return {
            "added": 0,
            "total": count_before,
            "error": "All searches failed - check Tavily API key.",
        }

    print("\nExtracting and structuring new behavioural questions...")
    new_questions = extract_behavioural_from_search(raw_text, existing)

    if not new_questions:
        return {
            "added": 0,
            "total": count_before,
            "error": "LLm extraction returned no questions",
        }

    unique_new = deduplicate_behavioural(existing, new_questions)
    # fix: assign IDs to the *deduplicated* list — previously this passed
    # new_questions, discarding the dedup result and re-adding duplicates.
    unique_new = assign_behavioural_ids(existing, unique_new)

    updated = existing + unique_new
    save_behavioural(updated)

    # Grouping new additions by competency for the summary
    by_competency = {}

    for q in unique_new:
        comp = q.get("competency", "unknown")
        by_competency.setdefault(comp, []).append(q["question"])

    return {
        "added": len(unique_new),
        "total": len(updated),
        "by_competency": by_competency,
        "error": None,
    }
