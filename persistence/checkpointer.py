import sqlite3
from contextlib import closing

from langgraph.checkpoint.sqlite import SqliteSaver


def get_checkpointer():
    """
    Returns a SQLite checkpointer that persists agent state across sessions.
    The DB file is created in the project root on first run.
    Uses closing() to ensure the connection is released when the checkpointer
    goes out of scope, preventing connection leaks across sessions.  # fix: connection was never closed
    """
    conn = sqlite3.connect("prep_agent.db", check_same_thread=False)
    return SqliteSaver(closing(conn))  # fix: wrap in closing() so connection is released on scope exit
