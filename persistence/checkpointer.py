import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver


def get_checkpointer():
    """
    Returns a SQLite checkpointer that persists agent state across sessions.
    The DB file is created in the project root on first run.
    """
    conn = sqlite3.connect("prep_agent.db", check_same_thread=False)
    return SqliteSaver(conn)
