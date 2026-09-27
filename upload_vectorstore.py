import faiss
import numpy as np


def create_upload_vectorstore(embedded_chunks):

    if not embedded_chunks:
        raise ValueError(
            "No embedded chunks were provided."
        )

    dimension = len(
        embedded_chunks[0]["embedding"]
    )

    index = faiss.IndexFlatL2(
        dimension
    )

    vectors = np.array(
        [
            chunk["embedding"]
            for chunk in embedded_chunks
        ],
        dtype="float32"
    )

    index.add(vectors)

    mapping = {
        str(index_id): {
            "chunk_id": chunk["chunk_id"],
            "page_content": chunk["page_content"],
            "metadata": chunk["metadata"]
        }
        for index_id, chunk in enumerate(embedded_chunks)
    }

    return index, mapping