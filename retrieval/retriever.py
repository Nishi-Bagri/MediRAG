from sentence_transformers import SentenceTransformer
import faiss
import json


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_FILE = "data/processed/vectorstore/index.faiss"
MAPPING_FILE = "data/processed/vectorstore/chunk_mapping.json"


def load_model():
    return SentenceTransformer(MODEL_NAME)


def load_vectorstore():
    index = faiss.read_index(INDEX_FILE)

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    return index, mapping


def retrieve(query, model, index, mapping, top_k=5):

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
        zip(distances[0], indices[0]),
        start=1
    ):
        result = mapping[str(index_id)]

        results.append({
            "rank": rank,
            "chunk_id": result["chunk_id"],
            "page_content": result["page_content"],
            "metadata": result["metadata"],
            "distance": float(distance)
        })

    return results