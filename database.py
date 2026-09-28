import sqlite3
import json

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

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_FILE,
        check_same_thread=False
    )

    connection.row_factory = sqlite3.Row

    # Enable foreign keys
    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()


    # --------------------------------------------------------
    # CONVERSATIONS
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )


    # --------------------------------------------------------
    # MESSAGES
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_data BLOB NOT NULL,
            uploaded_at TEXT NOT NULL
        )
        """
    )


    # --------------------------------------------------------
    # DOCUMENT CHUNKS
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            document_id INTEGER NOT NULL,

            chunk_id TEXT NOT NULL,

            page INTEGER,

            content TEXT NOT NULL,

            embedding TEXT,

            FOREIGN KEY (document_id)
                REFERENCES documents(id)
                ON DELETE CASCADE
        )
        """
    )


    connection.commit()

    connection.close()


# ============================================================
# DOCUMENT FUNCTIONS
# ============================================================

def save_document(
    filename,
    file_data
):
    """
    Permanently save an uploaded PDF in the database.

    Returns:
        document_id
    """

    connection = get_connection()

    cursor = connection.cursor()

    uploaded_at = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO documents
        (
            filename,
            file_data,
            uploaded_at
        )
        VALUES (?, ?, ?)
        """,
        (
            filename,
            sqlite3.Binary(file_data),
            uploaded_at
        )
    )

    document_id = cursor.lastrowid

    connection.commit()

    connection.close()

    return document_id


# ============================================================
# SAVE DOCUMENT CHUNKS
# ============================================================

def save_document_chunks(
    document_id,
    chunks
):
    """
    Save processed chunks and embeddings
    belonging to a document.
    """

    connection = get_connection()

    cursor = connection.cursor()


    for chunk in chunks:

        embedding = chunk.get(
            "embedding"
        )

        # Convert embedding list to JSON
        if embedding is not None:

            embedding = json.dumps(
                embedding
            )


        metadata = chunk.get(
            "metadata",
            {}
        )

        page = metadata.get(
            "page"
        )


        cursor.execute(
            """
            INSERT INTO document_chunks
            (
                document_id,
                chunk_id,
                page,
                content,
                embedding
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                document_id,
                chunk["chunk_id"],
                page,
                chunk["page_content"],
                embedding
            )
        )


    connection.commit()

    connection.close()


# ============================================================
# GET ALL DOCUMENTS
# ============================================================

def get_documents():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            filename,
            uploaded_at
        FROM documents
        ORDER BY uploaded_at DESC
        """
    )

    documents = cursor.fetchall()

    connection.close()

    return documents


# ============================================================
# GET DOCUMENT
# ============================================================

def get_document(
    document_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            filename,
            file_data,
            uploaded_at
        FROM documents
        WHERE id = ?
        """,
        (
            document_id,
        )
    )

    document = cursor.fetchone()

    connection.close()

    return document


# ============================================================
# GET DOCUMENT CHUNKS
# ============================================================

def get_document_chunks(
    document_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            document_id,
            chunk_id,
            page,
            content,
            embedding
        FROM document_chunks
        WHERE document_id = ?
        ORDER BY id ASC
        """,
        (
            document_id,
        )
    )

    rows = cursor.fetchall()

    connection.close()


    chunks = []


    for row in rows:

        embedding = None

        if row["embedding"]:

            try:

                embedding = json.loads(
                    row["embedding"]
                )

            except json.JSONDecodeError:

                embedding = None


        chunks.append(
            {
                "chunk_id": row["chunk_id"],

                "page_content": row["content"],

                "metadata": {
                    "page": row["page"],
                    "document_id": row["document_id"]
                },

                "embedding": embedding
            }
        )


    return chunks


# ============================================================
# DELETE DOCUMENT
# ============================================================

def delete_document(
    document_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM documents
        WHERE id = ?
        """,
        (
            document_id,
        )
    )

    connection.commit()

    connection.close()


# ============================================================
# CREATE CONVERSATION
# ============================================================

def create_conversation(
    title="New Chat"
):

    connection = get_connection()

    cursor = connection.cursor()

    created_at = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO conversations
        (
            title,
            created_at
        )
        VALUES (?, ?)
        """,
        (
            title,
            created_at
        )
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
        SELECT
            id,
            title,
            created_at
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

def get_messages(
    conversation_id
):

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
        (
            conversation_id,
        )
    )

    rows = cursor.fetchall()

    connection.close()


    messages = []


    for row in rows:

        sources = []


        if row["sources"]:

            try:

                sources = json.loads(
                    row["sources"]
                )

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

def delete_conversation(
    conversation_id
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        DELETE FROM messages
        WHERE conversation_id = ?
        """,
        (
            conversation_id,
        )
    )


    cursor.execute(
        """
        DELETE FROM conversations
        WHERE id = ?
        """,
        (
            conversation_id,
        )
    )


    connection.commit()

    connection.close()