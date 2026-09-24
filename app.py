import streamlit as st

from rag_pipeline import rag_pipeline

from database import (
    initialize_database,
    create_conversation,
    get_conversations,
    get_messages,
    save_message,
    update_conversation_title,
    delete_conversation
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediRAG",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

initialize_database()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .disclaimer {
        padding: 14px 18px;
        border-radius: 10px;
        background-color: #fff8e1;
        border: 1px solid #f0d98c;
        color: #5f4b00;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .answer-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .example-title {
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def start_new_conversation():

    conversation_id = create_conversation(
        "New Chat"
    )

    st.session_state.conversation_id = conversation_id
    st.session_state.messages = []


def ensure_conversation(title):

    if st.session_state.conversation_id is None:

        conversation_id = create_conversation(
            title
        )

        st.session_state.conversation_id = conversation_id

    else:

        update_conversation_title(
            st.session_state.conversation_id,
            title
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🩺 MediRAG")

    st.markdown(
        """
        **Medical Retrieval-Augmented Generation**

        MediRAG answers questions using information retrieved
        from its configured medical knowledge base.
        """
    )

    st.divider()


    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "💬 New Chat",
        use_container_width=True
    ):

        start_new_conversation()

        st.rerun()


    # --------------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------------

    st.subheader("🕘 History")

    conversations = get_conversations()

    if conversations:

        for conversation in conversations:

            conversation_id = conversation["id"]
            title = conversation["title"]

            if title == "New Chat":
                display_title = "New conversation"
            else:
                display_title = title


            # ------------------------------------------------
            # HISTORY ROW
            # ------------------------------------------------

            history_col, delete_col = st.columns(
                [5, 1],
                gap="small"
            )


            # ------------------------------------------------
            # OPEN CONVERSATION
            # ------------------------------------------------

            with history_col:

                if st.button(
                    display_title,
                    key=f"conversation_{conversation_id}",
                    use_container_width=True
                ):

                    st.session_state.conversation_id = conversation_id

                    st.session_state.messages = get_messages(
                        conversation_id
                    )

                    st.rerun()


            # ------------------------------------------------
            # DELETE CONVERSATION
            # ------------------------------------------------

            with delete_col:

                if st.button(
                    "🗑️",
                    key=f"delete_{conversation_id}",
                    help="Delete this conversation"
                ):

                    delete_conversation(
                        conversation_id
                    )

                    # If deleting the currently open chat,
                    # clear the current session.
                    if (
                        st.session_state.conversation_id
                        == conversation_id
                    ):

                        st.session_state.conversation_id = None
                        st.session_state.messages = []

                    st.rerun()

    else:

        st.caption("No conversations yet.")


    st.divider()


    # --------------------------------------------------------
    # SYSTEM ARCHITECTURE
    # --------------------------------------------------------

    with st.expander("⚙️ System Architecture"):

    st.markdown(
        """
        **MediRAG Pipeline**

        PDF<br>
        ↓<br>
        Ingestion<br>
        ↓<br>
        Cleaning<br>
        ↓<br>
        Chunking<br>
        ↓<br>
        Embeddings<br>
        ↓<br>
        FAISS Vector Database<br>
        ↓<br>
        Retrieval<br>
        ↓<br>
        LLM Generation
        """,
        unsafe_allow_html=True
    )


    st.divider()


    # --------------------------------------------------------
    # CLEAR CURRENT CHAT
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Current Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.conversation_id = None

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🩺 MediRAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Retrieval-Augmented Generation for grounded medical information'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
    <strong>Educational Project:</strong>
    MediRAG uses a synthetic medical knowledge base for RAG
    experimentation. It is not a diagnostic tool, does not provide
    medical diagnosis or treatment recommendations, and should not
    replace professional medical advice.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown(
    '<div class="example-title">Try an example question</div>',
    unsafe_allow_html=True
)


example_questions = [
    "What is HbA1c?",
    "What is the role of insulin?",
    "What factors can influence blood glucose?",
    "What are some commonly discussed symptoms of elevated blood glucose?"
]


example_columns = st.columns(2)


for index, question in enumerate(example_questions):

    with example_columns[index % 2]:

        if st.button(
            question,
            key=f"example_{index}",
            use_container_width=True
        ):

            st.session_state.pending_question = question

            st.rerun()


# ============================================================
# PROCESS EXAMPLE QUESTION
# ============================================================

if "pending_question" in st.session_state:

    pending_question = st.session_state.pop(
        "pending_question"
    )


    ensure_conversation(
        pending_question[:40]
    )


    st.session_state.messages.append(
        {
            "role": "user",
            "content": pending_question
        }
    )


    save_message(
        conversation_id=st.session_state.conversation_id,
        role="user",
        content=pending_question
    )


    with st.spinner(
        "Searching the knowledge base and generating an answer..."
    ):

        try:

            result = rag_pipeline(
                pending_question,
                return_sources=True
            )

            answer = result["answer"]

            sources = result.get(
                "sources",
                []
            )


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )


            save_message(
                conversation_id=st.session_state.conversation_id,
                role="assistant",
                content=answer,
                sources=sources
            )


        except Exception:

            error_message = (
                "Sorry, an error occurred while processing "
                "your question. Please try again."
            )


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                }
            )


            save_message(
                conversation_id=st.session_state.conversation_id,
                role="assistant",
                content=error_message
            )


    st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


        sources = message.get(
            "sources",
            []
        )


        if (
            message["role"] == "assistant"
            and sources
        ):

            with st.expander("📚 Sources"):

                for source in sources:

                    page = source.get(
                        "page",
                        "Unknown"
                    )


                    chunk_id = source.get(
                        "chunk_id",
                        "Unknown"
                    )


                    st.markdown(
                        f"- Page {page} · `{chunk_id}`"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask a question about the medical knowledge base..."
)


if query:

    ensure_conversation(
        query[:40]
    )


    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    save_message(
        conversation_id=st.session_state.conversation_id,
        role="user",
        content=query
    )


    with st.chat_message("user"):

        st.markdown(
            query
        )


    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the knowledge base and generating an answer..."
        ):

            try:

                result = rag_pipeline(
                    query,
                    return_sources=True
                )


                answer = result["answer"]

                sources = result.get(
                    "sources",
                    []
                )


                st.markdown(
                    answer
                )


                if sources:

                    with st.expander("📚 Sources"):

                        for source in sources:

                            page = source.get(
                                "page",
                                "Unknown"
                            )


                            chunk_id = source.get(
                                "chunk_id",
                                "Unknown"
                            )


                            st.markdown(
                                f"- Page {page} · `{chunk_id}`"
                            )


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )


                save_message(
                    conversation_id=st.session_state.conversation_id,
                    role="assistant",
                    content=answer,
                    sources=sources
                )


            except Exception:

                error_message = (
                    "Sorry, an error occurred while processing "
                    "your question. Please try again."
                )


                st.error(
                    error_message
                )


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )


                save_message(
                    conversation_id=st.session_state.conversation_id,
                    role="assistant",
                    content=error_message
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MediRAG • Educational RAG Project • Synthetic Medical Knowledge Base"
)