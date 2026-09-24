import logging

from logging_config import setup_logging


from retrieval.retriever import (
    load_model,
    load_vectorstore,
    retrieve
)

from generation.generator import (
    create_client,
    build_context,
    build_prompt,
    generate_answer
)

# ============================================================
# LOGGING
# ============================================================

setup_logging()

logger = logging.getLogger(__name__)


# ============================================================
# RAG PIPELINE
# ============================================================

def rag_pipeline(
    query,
    return_sources=False
):
    """
    Execute the complete MediRAG pipeline.

    Pipeline:
        Query
        ↓
        Embedding
        ↓
        FAISS Retrieval
        ↓
        Relevance Filtering
        ↓
        Context Construction
        ↓
        Prompt Construction
        ↓
        LLM Generation
    """

    if not query or not query.strip():

        logger.warning(
            "Empty query received."
        )

        message = (
            "Please enter a question about the "
            "medical knowledge base."
        )

        if return_sources:

            return {
                "answer": message,
                "sources": []
            }

        return message


    query = query.strip()


    logger.info(
        "RAG query received: %s",
        query
    )


    try:

        # ----------------------------------------------------
        # LOAD RETRIEVAL COMPONENTS
        # ----------------------------------------------------

        model = load_model()

        index, mapping = load_vectorstore()


        # ----------------------------------------------------
        # RETRIEVE RELEVANT CHUNKS
        # ----------------------------------------------------

        results = retrieve(
            query=query,
            model=model,
            index=index,
            mapping=mapping,
            top_k=5,
            max_distance=1.5
        )


        logger.info(
            "Retrieved %d relevant chunks.",
            len(results)
        )


        # ----------------------------------------------------
        # NO RELEVANT INFORMATION
        # ----------------------------------------------------

        if not results:

            logger.info(
                "No relevant chunks found for query."
            )

            message = (
                "The provided knowledge base does not contain "
                "enough relevant information to answer this question."
            )

            if return_sources:

                return {
                    "answer": message,
                    "sources": []
                }

            return message


        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context = build_context(
            results
        )


        # ----------------------------------------------------
        # BUILD PROMPT
        # ----------------------------------------------------

        prompt = build_prompt(
            context,
            query
        )


        # ----------------------------------------------------
        # CREATE LLM CLIENT
        # ----------------------------------------------------

        client = create_client()


        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        answer = generate_answer(
            client,
            prompt
        )


        logger.info(
            "Answer generated successfully."
        )


        # ----------------------------------------------------
        # RETURN ANSWER + SOURCES
        # ----------------------------------------------------

        if return_sources:

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


        return answer


    except Exception:

        logger.exception(
            "RAG pipeline failed."
        )


        error_message = (
            "Sorry, an internal error occurred while "
            "processing your question. Please try again."
        )


        if return_sources:

            return {
                "answer": error_message,
                "sources": []
            }


        return error_message