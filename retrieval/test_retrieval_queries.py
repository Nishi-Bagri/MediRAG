from retriever import load_model, load_vectorstore, retrieve


TEST_QUERIES = [
    "What is diabetes?",
    "What are the symptoms of diabetes?",
    "What is HbA1c?",
    "What is hypertension?",
    "What is blood glucose?",
    "What is hypoglycemia?",
    "What is hyperglycemia?",
    "What are cardiovascular risk factors?",
    "What is obesity?",
    "What is RAG?"
]


print("===== MEDIRAG RETRIEVAL QUERY TEST =====")

model = load_model()
index, mapping = load_vectorstore()


for query in TEST_QUERIES:

    results = retrieve(
        query,
        model,
        index,
        mapping,
        top_k=3
    )

    print("\nQuery:", query)

    for result in results:

        print(
            f"Rank {result['rank']} | "
            f"{result['chunk_id']} | "
            f"Distance: {result['distance']:.4f}"
        )