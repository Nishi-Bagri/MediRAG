import streamlit as st

from rag_pipeline import rag_pipeline

from upload_processor import process_uploaded_pdf
from upload_embeddings import embed_uploaded_chunks
from upload_vectorstore import create_upload_vectorstore

from retrieval.retriever import load_model, retrieve

from generation.generator import (
    create_client,
    build_context,
    build_prompt,
    generate_answer
)


from database import (
    initialize_database,
    create_conversation,
    get_conversations,
    get_messages,
    save_message,
    update_conversation_title,
    delete_conversation,
    get_document_chunks,

    # Document management
    save_document,
    save_document_chunks,
    get_documents,
    get_document,
    delete_document
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

    /* ---------- Global palette ---------- */
    :root {
        --mr-primary: #0d9488;      /* teal */
        --mr-primary-dark: #0f766e;
        --mr-accent: #6366f1;       /* indigo */
        --mr-bg-soft: #f4faf9;
        --mr-text-muted: #64748b;
        --mr-border: #e2e8f0;
    }

    /* ---------- App background ---------- */
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #f4faf9 100%);
    }

    /* ---------- Header ---------- */
    .main-title {
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 0;
        background: linear-gradient(90deg, var(--mr-primary), var(--mr-accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .subtitle {
        font-size: 16px;
        color: var(--mr-text-muted);
        margin-top: 6px;
        margin-bottom: 22px;
    }

    /* ---------- Disclaimer banner ---------- */
    .disclaimer {
        padding: 16px 20px;
        border-radius: 14px;
        background: linear-gradient(135deg, #fff8e1 0%, #fef3c7 100%);
        border: 1px solid #f3d98c;
        color: #6b4e00;
        font-size: 14px;
        margin-bottom: 28px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .answer-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 10px;
        color: #0f172a;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0fdfa 100%);
        border-right: 1px solid var(--mr-border);
    }

    section[data-testid="stSidebar"] h1 {
        color: var(--mr-primary-dark);
        font-weight: 800;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--mr-border);
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }

    .stButton > button:hover {
        border-color: var(--mr-primary);
        color: var(--mr-primary-dark);
        box-shadow: 0 2px 6px rgba(13, 148, 136, 0.15);
    }

    /* Primary "New Chat" style button (first sidebar button) */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #ffffff;
    }

    /* ---------- Chat bubbles ---------- */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 4px 6px;
        margin-bottom: 6px;
    }

    /* ---------- Chat input ---------- */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
    }

    /* ---------- Expander (Sources) ---------- */
    details {
        border-radius: 10px;
        border: 1px solid var(--mr-border) !important;
        background-color: #fafafa;
    }

    /* ---------- Misc ---------- */
    hr {
        margin: 1.2rem 0;
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


# ------------------------------------------------------------
# KNOWLEDGE SOURCE
# ------------------------------------------------------------

if "active_source_type" not in st.session_state:
    st.session_state.active_source_type = "none"

if "active_document_id" not in st.session_state:
    st.session_state.active_document_id = None


# ------------------------------------------------------------
# UPLOADED DOCUMENT STATE
# ------------------------------------------------------------

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "uploaded_index" not in st.session_state:
    st.session_state.uploaded_index = None

if "uploaded_mapping" not in st.session_state:
    st.session_state.uploaded_mapping = None

if "uploaded_chunk_count" not in st.session_state:
    st.session_state.uploaded_chunk_count = 0

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


def run_uploaded_rag(query):

    model = load_model()

    results = retrieve(
        query=query,
        model=model,
        index=st.session_state.uploaded_index,
        mapping=st.session_state.uploaded_mapping,
        top_k=5,
        max_distance=1.5
    )

    if not results:

        return {
            "answer": (
                "The uploaded document does not contain "
                "enough relevant information to answer "
                "this question."
            ),
            "sources": []
        }

    context = build_context(
        results
    )

    prompt = build_prompt(
        context,
        query
    )

    client = create_client()

    answer = generate_answer(
        client,
        prompt
    )

    sources = []

    for result in results:

        metadata = result.get(
            "metadata",
            {}
        )

        sources.append(
            {
                "rank": result["rank"],
                "chunk_id": result["chunk_id"],
                "page": metadata.get(
                    "page",
                    "Unknown"
                ),
                "distance": result["distance"]
            }
        )

    return {
        "answer": answer,
        "sources": sources
    }
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

    # ============================================================
    # KNOWLEDGE SOURCE
    # ============================================================

    st.markdown("### 📚 Knowledge Source")

    documents = get_documents()

    source_options = {
        "none": "⚪ Select a knowledge source",
        "default": "🔵 Default Knowledge Base"
    }

    for document in documents:

        source_options[
            f"document_{document['id']}"
        ] = f"📄 {document['filename']}"


    # ------------------------------------------------------------
    # INITIALIZE RADIO STATE ONLY ONCE
    # ------------------------------------------------------------

    if "knowledge_source_selector" not in st.session_state:

        st.session_state.knowledge_source_selector = "none"


    # ------------------------------------------------------------
    # REMOVE DELETED DOCUMENT FROM SELECTION
    # ------------------------------------------------------------

    current_selection = (
        st.session_state.knowledge_source_selector
    )

    if current_selection not in source_options:

        st.session_state.knowledge_source_selector = "none"

        st.session_state.active_source_type = "none"
        st.session_state.active_document_id = None


    # ------------------------------------------------------------
    # KNOWLEDGE SOURCE RADIO
    # ------------------------------------------------------------

    selected_source = st.radio(
        "Select knowledge source",
        options=list(source_options.keys()),
        format_func=lambda x: source_options[x],
        key="knowledge_source_selector"
    )


    # ------------------------------------------------------------
    # UPDATE ACTIVE SOURCE
    # ------------------------------------------------------------

    if selected_source == "none":

        st.session_state.active_source_type = "none"
        st.session_state.active_document_id = None

        st.caption(
            "Select a knowledge source before asking a question."
        )


    elif selected_source == "default":

        st.session_state.active_source_type = "default"
        st.session_state.active_document_id = None

        st.caption(
            "Using the default MediRAG medical knowledge base."
        )


    elif selected_source.startswith("document_"):
        document_id = int(
            selected_source.split("_")[1]
        )

        st.session_state.active_source_type = "document"
        st.session_state.active_document_id = document_id

        selected_document = next(
            (
                document
                for document in documents
                if document["id"] == document_id
            ),
            None
        )

        if selected_document:

            st.caption(
                f"Using: {selected_document['filename']}"
            )

            # LOAD SAVED DOCUMENT
            if (
                st.session_state.get("uploaded_document_id")
                != document_id
            ):

                try:

                    saved_chunks = get_document_chunks(
                        document_id
                    )

                    if not saved_chunks:

                        st.warning(
                            "No saved chunks found for this document."
                        )

                        st.session_state.uploaded_index = None
                        st.session_state.uploaded_mapping = None

                    else:

                        index, mapping = create_upload_vectorstore(
                            saved_chunks
                        )

                        st.session_state.uploaded_index = index
                        st.session_state.uploaded_mapping = mapping

                        st.session_state.uploaded_document_id = (
                            document_id
                        )

                        st.session_state.uploaded_file_name = (
                            selected_document["filename"]
                        )

                        st.session_state.uploaded_chunk_count = (
                            len(saved_chunks)
                        )

                except Exception as e:

                    st.error(
                        f"Failed to load saved document: {e}"
                    )

                    st.session_state.uploaded_index = None
                    st.session_state.uploaded_mapping = None

    st.divider()

    # ============================================================
    # UPLOAD DOCUMENT
    # ============================================================

    uploaded_file = st.file_uploader(
        "📄 Upload a PDF",
        type=["pdf"],
        help="Upload a PDF to permanently save it as a knowledge source."
    )

    if uploaded_file is not None:

        # Process only when a new file is uploaded
        if (
            st.session_state.uploaded_file_name
            != uploaded_file.name
        ):

            with st.spinner(
                "Processing and saving PDF..."
            ):

                try:

                    pdf_bytes = uploaded_file.getvalue()

                    document_id = save_document(
                        uploaded_file.name,
                        pdf_bytes
                    )

                    chunks = process_uploaded_pdf(
                        pdf_bytes,
                        uploaded_file.name
                    )

                    embedded_chunks = embed_uploaded_chunks(
                        chunks
                    )

                    save_document_chunks(
                        document_id,
                        embedded_chunks
                    )

                    index, mapping = create_upload_vectorstore(
                        embedded_chunks
                    )

                    st.session_state.uploaded_file_name = (
                        uploaded_file.name
                    )

                    st.session_state.uploaded_document_id = (
                        document_id
                    )

                    st.session_state.uploaded_index = index

                    st.session_state.uploaded_mapping = mapping

                    st.session_state.uploaded_chunk_count = (
                        len(chunks)
                    )

                    st.success(
                        f"PDF saved successfully — "
                        f"{len(chunks)} chunks"
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"PDF processing failed: {e}"
                    )


        # --------------------------------------------------------
        # ACTIVE DOCUMENT
        # --------------------------------------------------------

        if (
            st.session_state.uploaded_file_name
            == uploaded_file.name
        ):

            chunk_count = st.session_state.get(
                "uploaded_chunk_count",
                0
            )

            st.info(
                f"📄 {uploaded_file.name} | "
                f"{chunk_count} chunks | "
                f"Active knowledge source"
            )


    # ============================================================
    # SAVED DOCUMENTS
    # ============================================================

    st.markdown("### 📚 Saved Documents")

    documents = get_documents()

    if documents:

        for document in documents:

            document_id = document["id"]
            filename = document["filename"]

            col1, col2 = st.columns([5, 1])

            with col1:

                st.markdown(
                    f"📄 **{filename}**"
                )

                st.caption(
                    "Permanently saved in database"
                )

            with col2:

                if st.button(
                    "🗑️",
                    key=f"delete_document_{document_id}",
                    help=f"Delete {filename}"
                ):

                    delete_document(document_id)

                    if (
                        st.session_state.get(
                            "uploaded_document_id"
                        )
                        == document_id
                    ):

                        st.session_state.uploaded_document_id = None
                        st.session_state.uploaded_file_name = None
                        st.session_state.uploaded_index = None
                        st.session_state.uploaded_mapping = None
                        st.session_state.uploaded_chunk_count = 0

                    st.rerun()

    else:

        st.caption("No saved documents.")


    # ============================================================
    # CONVERSATION HISTORY
    # ============================================================

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

            history_col, delete_col = st.columns(
                [5, 1],
                gap="small"
            )

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

            with delete_col:

                if st.button(
                    "🗑️",
                    key=f"delete_{conversation_id}",
                    help="Delete this conversation"
                ):

                    delete_conversation(
                        conversation_id
                    )

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

    # ========================================================
    # CHECK KNOWLEDGE SOURCE
    # ========================================================

    if st.session_state.active_source_type == "none":

        st.warning(
            "Please select a knowledge source before asking a question."
        )

    else:

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

                    # ====================================================
                    # DEFAULT KNOWLEDGE BASE
                    # ====================================================

                    if (
                        st.session_state.active_source_type
                        == "default"
                    ):

                        result = rag_pipeline(
                            query,
                            return_sources=True
                        )

                    # ====================================================
                    # UPLOADED DOCUMENT
                    # ====================================================

                    elif (
                        st.session_state.active_source_type
                        == "document"
                    ):

                        if (
                            st.session_state.uploaded_index is None
                            or
                            st.session_state.uploaded_mapping is None
                        ):

                            st.error(
                                "The selected document is not currently loaded. "
                                "Please upload or reload the document."
                            )

                            st.stop()

                        result = run_uploaded_rag(
                            query
                        )

                    # ====================================================
                    # INVALID SOURCE
                    # ====================================================

                    else:

                        st.warning(
                            "Please select a valid knowledge source."
                        )

                        st.stop()


                    # ====================================================
                    # ANSWER
                    # ====================================================

                    answer = result["answer"]

                    sources = result.get(
                        "sources",
                        []
                    )

                    st.markdown(
                        answer
                    )


                    # ====================================================
                    # SOURCES
                    # ====================================================

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


                    # ====================================================
                    # SAVE ASSISTANT MESSAGE
                    # ====================================================

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
                        content=error_message,
                        sources=[]
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MediRAG • Educational RAG Project • Synthetic Medical Knowledge Base"
)