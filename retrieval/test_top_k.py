from retriever import load_model, load_vectorstore, retrieve


print("===== TOP-K RETRIEVAL TEST =====")

model = load_model()

index, mapping = load_vectorstore()

query = "What is diabetes?"

for k in [1, 3, 5]:

    results = retrieve(
        query,
        model,
        index,
        mapping,
        top_k=k
    )

    print(f"\nTop-K = {k}")
    print("Results returned:", len(results))

    for result in results:
        print(
            result["rank"],
            result["chunk_id"],
            result["distance"]
        )