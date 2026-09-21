from retriever import load_model, load_vectorstore, retrieve


print("===== OUT-OF-SCOPE RETRIEVAL TEST =====")

model = load_model()
index, mapping = load_vectorstore()

queries = [
    "What is the capital of France?",
    "How do I repair a car engine?",
    "What is the population of Japan?"
]

for query in queries:

    results = retrieve(
        query,
        model,
        index,
        mapping,
        top_k=5
    )

    print(f"\nQuery: {query}")
    print("Results returned:", len(results))

    for result in results[:3]:
        print(
            "Rank:",
            result["rank"],
            "| Chunk:",
            result["chunk_id"],
            "| Distance:",
            result["distance"]
        )