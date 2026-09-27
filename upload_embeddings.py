from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def embed_uploaded_chunks(chunks):
    """ 
    Generate emdeddings for chunk from the currently uploaded PDF.
    """

    if not chunks:
        raise ValueError(
            "No chunks were provided."
        )

    model = SentenceTransformer(
        MODEL_NAME
    )

    embedded_chunks = []

    for chunk in chunks:

        embedding = model.encode(
            chunk["page_content"]
        )

        embedded_chunk = {
            "chunk_id": chunk["chunk_id"],
            "page_content": chunk["page_content"],
            "metadata": chunk["metadata"],
            "embedding": embedding.tolist()
        }

        embedded_chunks.append(
            embedded_chunk
        )

    return embedded_chunks

