from retriever import load_model, load_vectorstore, retrieve


TEST_CASES = [
    {
        "query": "What is diabetes?",
        "expected_chunk": "page_7_chunk_1"
    },
    {
        "query": "What are the symptoms of diabetes?",
        "expected_chunk": "page_7_chunk_2"
    },
    {
        "query": "What is HbA1c?",
        "expected_chunk": "page_7_chunk_3"
    },
    {
        "query": "What is hypertension?",
        "expected_chunk": "page_9_chunk_1"
    }
]


print("===== MRR EVALUATION =====")

model = load_model()
index, mapping = load_vectorstore()

reciprocal_ranks = []

for test in TEST_CASES:

    results = retrieve(
        test["query"],
        model,
        index,
        mapping,
        top_k=5
    )

    reciprocal_rank = 0.0

    for result in results:

        if result["chunk_id"] == test["expected_chunk"]:
            reciprocal_rank = 1 / result["rank"]
            break

    reciprocal_ranks.append(reciprocal_rank)

    print(f"\nQuery: {test['query']}")
    print("Reciprocal Rank:", reciprocal_rank)


mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

print("\n===== MRR SUMMARY =====")
print("Total queries:", len(TEST_CASES))
print("MRR:", mrr)