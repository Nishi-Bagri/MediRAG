from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

INDEX_FILE = "data/processed/vectorstore/index.faiss"
MAPPING_FILE = "data/processed/vectorstore/chunk_mapping.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if __name__ == "__main__":

    print("===== REAL QUERY VECTOR SEARCH =====")
    
    model = SentenceTransformer(MODEL_NAME)

    index = faiss.read_index(INDEX_FILE)
    

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    query = "what is diabetes?"

    query_embedding = model.encode(
        [query],
        convert_to_numpy = True).astype("float32")


    print("Query:", query)
    print("Query vector dimension:", query_embedding.shape[1])

    distances, indices = index.search(query_embedding, 3)

    print("\n===== SEARCH RESULT =====")

    for rank, (distance, idx) in enumerate(
        zip(distances[0], indices[0]), start = 1
        ):

        result = mapping[str(idx)]

        print(f"\nRank: {rank}")
        print("Distance:", distance)
        print("Chunk ID:", result["chunk_id"])
        print("Page:", result["metadata"]["page"])