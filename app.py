import streamlit as st

from rag_pipeline import rag_pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediRAG",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)


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


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏥 MediRAG")

    st.markdown(
        """
        **Medical Retrieval-Augmented Generation**

        MediRAG answers questions using information retrieved
        from its configured medical knowledge base.
        """
    )

    st.divider()

    st.subheader("Project")

    st.markdown(
        """
        **Pipeline**

        PDF  
        ↓  
        Ingestion  
        ↓  
        Cleaning  
        ↓  
        Chunking  
        ↓  
        Embeddings  
        ↓  
        FAISS  
        ↓  
        Retrieval  
        ↓  
        LLM Generation
        """
    )

    st.divider()

    st.subheader("Evaluation")

    st.markdown(
        """
        - Recall@5: **100%**
        - MRR: **0.75**
        - Precision@5: **20%**
        - Generation: **100%**
        - Hallucination Safety: **100%**
        """
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏥 MediRAG</div>',
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

    pending_question = st.session_state.pop("pending_question")

    st.session_state.messages.append(
        {
            "role": "user",
            "content": pending_question
        }
    )

    with st.spinner("Searching the knowledge base and generating an answer..."):

        try:

            answer = rag_pipeline(pending_question)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception:

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Sorry, an error occurred while processing "
                        "your question. Please try again."
                    )
                }
            )

    st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask a question about the medical knowledge base..."
)


if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the knowledge base and generating an answer..."
        ):

            try:

                answer = rag_pipeline(query)

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception:

                error_message = (
                    "Sorry, an error occurred while processing "
                    "your question. Please try again."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MediRAG • Educational RAG Project • Synthetic Medical Knowledge Base"
)