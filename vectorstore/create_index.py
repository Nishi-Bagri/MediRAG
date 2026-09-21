import json
import faiss
import numpy as np


INPUT_FILE = "data/processed/embedded_chunks.json"
INDEX_FILE = "data/processed/vectorstore/index.faiss"


def load_embeddings():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def create_faiss_index(data):
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

    print("===== VECTOR DATABASE CREATION =====")
    print("Records loaded:", len(data))

    index = create_faiss_index(data)

    print("Vector dimension:", index.d)
    print("Vectors stored:", index.ntotal)

    faiss.write_index(index, INDEX_FILE)

    print("FAISS index saved:", INDEX_FILE)