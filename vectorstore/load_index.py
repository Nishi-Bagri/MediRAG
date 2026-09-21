import faiss

INDEX_FILE = "data/processed/vectorstore/index.faiss"

if __name__ == "__main__":
    index = faiss.read_index(INDEX_FILE)

    print("===== SAVED FAISS INDEX LOADED =====")
    print("Index trained:", index.is_trained)
    print("Vectors loaded:", index.ntotal)
    print("Vector dimension:", index.d)
