import os

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialising LLM at module level
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Examples the LLM uses as reference — few-shot prompting
EXAMPLES = """
"quiz me on networking"           → revise
"test me on TCP"                  → revise
"I want a question"               → revise
"teach me something"              → learn
"explain how DNS works"           → learn
"I want to learn about databases" → learn
"practise a behavioural question" → star
"STAR question please"            → star
"help me with an interview story" → star
"how am I doing"                  → progress
"show my progress"                → progress
"what are my weak topics"         → progress
"update the topic bank"           → refresh
"search for new topics"           → refresh
"refresh the question bank"       → refresh
"find new CS topics"              → refresh
"refresh behavioural questions"   → refresh_star
"update my STAR question bank"    → refresh_star
"find new behavioural questions"  → refresh_star
"add more competency questions"   → refresh_star
"""


def classify_intent(user_input: str) -> str:
    """
    Classifies the user's free-form input into on of four intent:
        lean | revise | star | progress | refresh | refresh_star
    """
    prompt = f"""You are an intent classifier for an interview prep agent.
    Classify the user's message into exactly one of these intents:
    - learn     → user wants to be taught a new topic
    - revise    → user wants to be quizzed / tested on a topic
    - star      → user wants to practise a behavioural/competency question
    - progress  → user wants to see their scores or progress summary
    - refresh   → user wants to search for and add new topics to the question bank
    - refresh_star  → user wants to update the behavioural question bank

    Examples:
    {EXAMPLES}

    User message: "{user_input}"

    Reply with a single word — one of: learn, revise, star, progress, refresh, refresh_star
    No explanation. No punctuation. Just the word."""

    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip().lower()

    # Validate - fall back to "revise" if LLM returns something unexpected
    valid = {"learn", "revise", "star", "progress", "refresh", "refresh_star"}
    return intent if intent in valid else "revise"
