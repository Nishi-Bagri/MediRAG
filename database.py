import sqlite3
from pathlib import Path
from datetime import datetime


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATABASE_FILE = DATA_DIR / "medirag.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_FILE,
        check_same_thread=False
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
                ON DELETE CASCADE
        )
        """
    )

    connection.commit()
    connection.close()


# ============================================================
# CREATE CONVERSATION
# ============================================================

def create_conversation(title="New Chat"):

    connection = get_connection()

    cursor = connection.cursor()

    created_at = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO conversations (title, created_at)
        VALUES (?, ?)
        """,
        (title, created_at)
    )

    conversation_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return conversation_id


# ============================================================
# GET ALL CONVERSATIONS
# ============================================================

def get_conversations():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, title, created_at
        FROM conversations
        ORDER BY created_at DESC
        """
    )

    conversations = cursor.fetchall()

    connection.close()

    return conversations


# ============================================================
# SAVE MESSAGE
# ============================================================

def save_message(
    conversation_id,
    role,
    content,
    sources=None
):

    import json

    connection = get_connection()

    cursor = connection.cursor()

    created_at = datetime.now().isoformat()

    sources_json = json.dumps(
        sources if sources else []
    )

    cursor.execute(
        """
        INSERT INTO messages
        (
            conversation_id,
            role,
            content,
            sources,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            conversation_id,
            role,
            content,
            sources_json,
            created_at
        )
    )

    connection.commit()
    connection.close()


# ============================================================
# GET CONVERSATION MESSAGES
# ============================================================

def get_messages(conversation_id):

    import json

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            role,
            content,
            sources,
            created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        (conversation_id,)
    )

    rows = cursor.fetchall()

    connection.close()

    messages = []

    for row in rows:

        sources = []

        if row["sources"]:

            try:
                sources = json.loads(row["sources"])

            except json.JSONDecodeError:
                sources = []

        messages.append(
            {
                "role": row["role"],
                "content": row["content"],
                "sources": sources,
                "created_at": row["created_at"]
            }
        )

    return messages


# ============================================================
# UPDATE CONVERSATION TITLE
# ============================================================

def update_conversation_title(
    conversation_id,
    title
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE conversations
        SET title = ?
        WHERE id = ?
        """,
        (
            title,
            conversation_id
        )
    )

    connection.commit()
    connection.close()


# ============================================================
# DELETE CONVERSATION
# ============================================================

def delete_conversation(conversation_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM messages
        WHERE conversation_id = ?
        """,
        (conversation_id,)
    )

    cursor.execute(
        """
        DELETE FROM conversations
        WHERE id = ?
        """,
        (conversation_id,)
    )

    connection.commit()
    connection.close()