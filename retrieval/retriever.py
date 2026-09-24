from functools import lru_cache

import faiss
import json
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_FILE = "data/processed/vectorstore/index.faiss"
MAPPING_FILE = "data/processed/vectorstore/chunk_mapping.json"


# ============================================================
# MODEL LOADING
# ============================================================

@lru_cache(maxsize=1)
def load_model():
    """
    Load the embedding model once and reuse it.
    """
    return SentenceTransformer(MODEL_NAME)


# ============================================================
# VECTOR STORE LOADING
# ============================================================

@lru_cache(maxsize=1)
def load_vectorstore():
    """
    Load the FAISS index and chunk mapping once and reuse them.
    """

    index = faiss.read_index(INDEX_FILE)

    with open(
        MAPPING_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        mapping = json.load(f)

    return index, mapping


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query,
    model,
    index,
    mapping,
    top_k=5,
    max_distance=1.5
):
    """
    Retrieve relevant chunks for a query.

    Results with an L2 distance greater than max_distance
    are treated as not sufficiently relevant.
    """

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (distance, index_id) in enumerate(
        zip(
            distances[0],
            indices[0]
        ),
        start=1
    ):

        if index_id == -1:
            continue

        result = mapping.get(
            str(index_id)
        )

        if result is None:
            continue

        distance = float(distance)

        # Relevance filtering
        if distance > max_distance:
            continue

        results.append(
            {
                "rank": rank,
                "chunk_id": result["chunk_id"],
                "page_content": result["page_content"],
                "metadata": result["metadata"],
                "distance": distance
            }
        )

    return results