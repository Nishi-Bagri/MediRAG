import json
import faiss
import numpy as np

INPUT_FILE = "data/processed/embedded_chunks.json"

def load_embeddings():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def create_index(data):
    embeddings = np.array(
        [item["embedding"] for item in data],
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index

if __name__ == "__main__":
    data = load_embeddings()
    index = create_index(data)

    print("===== FAISS INDEX VALIDATION =====")
    print("Index trained:", index.is_trained)
    print("Expected vectors:", len(data))
    print("Stored vectors:", index.ntotal)
    print("Expected dimension:", 384)
    print("Actual dimension:", index.d)

    if (
        index.is_trained
        and index.ntotal == len(data)
        and index.d == 384
    ):
        print("FAISS INDEX VALIDATION: PASS")
    else:
        print("FAISS INDEX VALIDATION: FAIL")
