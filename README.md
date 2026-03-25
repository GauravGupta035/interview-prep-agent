# Interview Prep Agent

A LangGraph-powered interview preparation agent with three modes: **CS learning**, **CS/Networking revision**, and **Behavioural STAR coaching**. Built to solve real daily pain points during UK tech job applications. Both question banks are kept current via web search using Tavily.

---

## What This Agent Does

| Mode | What Happens |
|---|---|
| **Learn** | Agent teaches a topic with concept, mechanism, analogy, and interview angle — then asks you to explain it back |
| **Revise** | Agent quizzes you on weak topics first, evaluates depth and accuracy, adapts based on your score history |
| **STAR** | Agent presents a UK competency question, scores your answer element by element, rewrites it in full STAR format |
| **Progress** | Shows your topic confidence scores, STAR session history, and competency coverage |
| **Refresh (CS)** | Searches the web for current UK interview topics and adds new ones to the CS question bank |
| **Refresh (STAR)** | Searches the web for current UK behavioural questions and adds new ones to the behavioural bank |

All modes share a **persistent learner profile** stored in SQLite — topic scores, revision history, and STAR coaching records accumulate across every session.

---

## Natural Language Entry Point

No menus. You just type what you want:

```
quiz me on networking
teach me something new
I want to practise a behavioural question
show my progress
refresh the topic bank
refresh behavioural questions
exit
```

An intent classifier (single LLM call with few-shot examples) routes your input to the correct mode. Falls back to `revise` if classification is ambiguous.

---

## High-Level Architecture

```
User Input
    ↓
[Intent Classifier]   ← single LLM call, few-shot prompted
    ↓
    ├── learn         → Learning Path
    ├── revise        → Revision Path
    ├── star          → STAR Coaching Path
    ├── progress      → Profile Summary
    ├── refresh       → CS Topic Bank Refresh
    └── refresh_star  → Behavioural Question Bank Refresh
```

---

## Mode 1 — Learning Path

```
[Select Topic Node]          ← new topics first, then weakest scored
    ↓
[Teach Node]                 ← concept / mechanism / analogy / interview angle
    ↓
[Comprehension Check Node]   ← "explain it back to me in 2-3 sentences"
    ↓
[interrupt_before]           ← waits for your explanation
    ↓
[Evaluate Understanding]     ← scores comprehension, identifies specific gap
    ↓
    ├── "understood"    → [Reinforce Node]  → affirms + suggests next step
    └── "needs_review"  → [Review Node]     → re-explains the exact gap missed
    ↓
[topics_introduced updated in profile]
```

**Topic selection logic:** Topics never introduced are picked first. Once all topics have been introduced at least once, the agent picks the lowest-scored topic.

**Teach node output format:**
- CONCEPT — what it is and why it matters
- HOW IT WORKS — key mechanism or process
- ANALOGY — one real-world analogy
- INTERVIEW ANGLE — how this typically comes up in UK interviews

---

## Mode 2 — Revision Path

```
[Select Topic Node]          ← prioritises topics with score < 0.5
    ↓
[Ask Question Node]          ← picks question type: conceptual / scenario / compare
    ↓
[interrupt_before]           ← waits for your answer
    ↓
[Evaluate Answer Node]       ← grades depth, accuracy, missing key points
    ↓
    ├── "strong"  → [Reinforce Node]  → affirms + adds depth
    └── "weak"    → [Explain Node]    → fills gaps, flags misconceptions
    ↓
[Update Profile Node]        ← moving average score saved to SQLite
```

**Scoring:** Each attempt updates the topic score as a moving average: `new_score = (existing + attempt) / 2`. Topics scoring below 0.5 are flagged as weak and surface more frequently.

**Question types per topic:**
- `conceptual` — "What is X and how does it work?"
- `scenario` — "You're building Y. How would you approach Z?"
- `compare` — "A colleague says X. How would you respond?"

---

## Mode 3 — STAR Coaching Path

```
[Present Question Node]      ← picks underrepresented competencies first
    ↓
[interrupt_before]           ← you give your raw answer
    ↓
[STAR Analyser Node]         ← structured JSON extraction of STAR elements
    ↓
[Critique Node]              ← element-by-element feedback with score bar
    ↓
[Rewrite Node]               ← full enhanced answer in STAR format
    ↓
    ├── retry  → back to [Present Question Node]
    └── done   → [Update Star Profile Node]
    ↓
[competencies_done + star_scores saved to SQLite]
```

### STAR Analyser — JSON Extraction Prompt

The analyser uses structured JSON extraction — not free-form feedback — so scores are comparable across sessions:

```
You are a UK competency interview coach.
Analyse the following answer for STAR structure.

Return ONLY valid JSON:
{
  "situation": {"present": bool, "quality": "strong/weak/missing", "feedback": "..."},
  "task":      {"present": bool, "quality": "strong/weak/missing", "feedback": "..."},
  "action":    {"present": bool, "quality": "strong/weak/missing", "feedback": "..."},
  "result":    {"present": bool, "quality": "strong/weak/missing", "feedback": "..."},
  "overall_score": 1-10,
  "biggest_gap": "...",
  "enhanced_version": "full rewritten answer in STAR format, 150-200 words"
}
```

**Action element flag:** The analyser specifically detects "we did" language and flags it — UK interviewers want to hear "I did."

**Result element flag:** Unquantified outcomes are flagged — "the system improved" is weak, "restored 30+ client systems within 2 hours" is strong.

### Critique Display

```
STAR Analysis  [████████░░]  8/10
────────────────────────────────────────────────────────────
S — Situation  [STRONG]
  Clear context established with timeline and stakes.

T — Task       [WEAK]
  Your specific responsibility was implied, not stated.

A — Action     [STRONG]
  Concrete personal actions with specific technologies named.

R — Result     [WEAK]
  Outcome mentioned but not quantified.

Biggest gap: Quantify the result — add numbers or time saved.
```

---

## Behavioural Competency Coverage

The agent tracks which competencies have been practised and prioritises untouched ones:

| Competency | Description |
|---|---|
| problem-solving | Structured approach to complex technical challenges |
| learning-agility | Picking up new skills quickly under pressure |
| stakeholder-management | Navigating difficult stakeholders, pushing back |
| leadership | Leading teams or initiatives through challenges |
| resilience | Handling mistakes, setbacks, and difficult feedback |
| delivery | Shipping under tight deadlines or limited resources |
| collaboration | Working across functions to deliver shared outcomes |
| initiative | Acting without being asked, identifying unseen problems |
| communication | Clarity across technical and non-technical audiences |
| adaptability | Changing direction when context shifts |
| ownership | Taking full accountability for outcomes |
| attention-to-detail | Catching issues others miss |

---

## Web Search — Topic Bank Refresh

Both question banks can be updated on demand via Tavily web search.

### CS/Networking Refresh

Search queries used:
```
"must know networking topics UK software engineer interviews 2026"
"common system design concepts UK tech interviews graduate"
"software engineering concepts asked in fintech UK interviews"
"CS fundamentals questions UK FAANG interviews 2026"
```

Process:
1. Four Tavily searches return clean pre-extracted text
2. LLM extracts structured topic entries in the exact JSON format
3. Deduplication by topic name (case-insensitive)
4. Sequential IDs assigned by category (e.g. `networking_006`)
5. `cs_concepts.json` updated in place

### Behavioural Refresh

Search queries used:
```
"UK competency based interview questions software engineer 2026"
"behavioural interview questions fintech UK"
"STAR format interview questions UK graduate software engineering"
"common behavioural interview questions UK tech firms 2026"
```

Additional logic: the LLM prompt receives a list of **underrepresented competencies** (those with fewer than 2 questions) and is instructed to prioritise those — so the bank fills gaps rather than adding more questions to already-covered competencies.

Deduplication uses a 60-character fingerprint of the question text to catch near-duplicates.

---

## Persistent Learner Profile

```python
class AgentState(TypedDict):
    # Conversation
    messages:           Annotated[list, operator.add]  # accumulates

    # Current session
    current_topic:      str
    current_question:   str
    user_answer:        str
    evaluation:         str
    verdict:            str

    # CS/Networking profile
    topic_scores:       dict    # {"TCP vs UDP": 0.8, "CAP theorem": 0.3}
    topics_weak:        list    # topics with score < 0.5
    topics_introduced:  list    # topics taught at least once

    # Behavioural profile
    current_competency: str
    star_analysis:      dict    # parsed JSON from STAR analyser
    star_scores:        list    # [{competency, date, score, gap}]
    competencies_done:  list    # competencies practised at least once
    retry_requested:    bool
```

Stored in **SQLite via LangGraph's `SqliteSaver` checkpointer**. Three thread IDs are used — `learn_main`, `revision_main`, `star_main` — so each mode's state is independently persisted but the progress summary aggregates across all three.

---

## Question Bank Structure

### `cs_concepts.json`

```json
{
  "id": "networking_001",
  "topic": "TCP vs UDP",
  "category": "networking",
  "difficulty": "medium",
  "question_types": [
    {"type": "conceptual", "q": "What is the key difference between TCP and UDP?"},
    {"type": "scenario",   "q": "You're building a live video streaming app. Which protocol would you choose and why?"},
    {"type": "compare",    "q": "A colleague says UDP is always faster than TCP. How would you respond?"}
  ],
  "key_points": ["reliability", "three-way handshake", "ordering", "use cases"],
  "common_mistakes": ["saying UDP is just faster without discussing tradeoffs"]
}
```

Categories: `networking`, `systems`, `databases`, `software_engineering`, `security`, `cloud`

### `behavioural_questions.json`

```json
{
  "id": "beh_001",
  "competency": "problem-solving",
  "question": "Tell me about a time you had to solve a complex technical problem with incomplete information.",
  "what_they_want": ["initiative", "structured thinking", "outcome focus"],
  "red_flags": ["blaming others", "no clear personal action", "vague outcome"]
}
```

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Agent framework | LangGraph | State machine control, `interrupt_before`, persistence built-in |
| Human-in-the-loop | `interrupt_before` | Pauses graph to wait for answer, resumes via `update_state` |
| Persistence | SQLite via `SqliteSaver` | Profile accumulates across all sessions |
| LLM | Gemini 2.5 Flash (Google) | Free tier, fast, capable for evaluation and coaching tasks |
| Web search | Tavily | Returns clean pre-extracted text — no HTML parsing needed |
| CLI | Plain Python | Runs in terminal, no frontend required |
| Coding assistant | Gemini API | Free equivalent to Claude Code for multi-file edits |

---

## Project Repository Structure

```
interview-prep-agent/
├── main.py                        # entry point — intent classifier + session runners
├── requirements.txt
├── .env                           # GOOGLE_API_KEY, TAVILY_API_KEY
├── prep_agent.db                  # SQLite — auto-created on first run
├── graph/
│   ├── __init__.py
│   ├── state.py                   # AgentState TypedDict
│   ├── router.py                  # conditional edge functions
│   └── nodes/
│       ├── __init__.py
│       ├── classifier.py          # intent classifier (LLM call)
│       ├── learn.py               # learning path nodes
│       ├── revise.py              # revision path nodes
│       └── star.py                # STAR coaching nodes
├── data/
│   ├── cs_concepts.json           # CS/networking question bank
│   ├── behavioural_questions.json # competency question bank
│   └── topic_refresher.py         # Tavily search + LLM extraction for both banks
└── persistence/
    └── checkpointer.py            # SqliteSaver setup
```
