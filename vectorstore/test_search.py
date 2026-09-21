import json 
import faiss
import numpy as np

INDEX_FILE = "data/processed/vectorstore/index.faiss"
MAPPING_FILE = "data/processed/vectorstore/chunk_mapping.json"
EMBEDDING_FILE = "data/processed/embedded_chunks.json"

if __name__ == "__main__":
    index = faiss.read_index(INDEX_FILE)

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    query_vector = np.array(
        [data[0]["embedding"]],
        dtype="float32"
    )

    distances, indices = index.search(query_vector, 3)

    print("===== VECTOR SEARCH TEST =====")

    for rank, (distance, idx) in enumerate(
        zip(distances[0], indices[0]), start=1
        ):

        result = mapping[str(idx)]

        print(f"\nRank: {rank}")
        print("FAISS position:", idx)
        print("Distance:", distance)
        print("Chunk ID:", result["chunk_id"])
        print("Print:", result["metadata"]["page"])