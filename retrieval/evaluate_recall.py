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


print("===== RECALL@5 EVALUATION =====")

model = load_model()
index, mapping = load_vectorstore()

hits = 0

for test in TEST_CASES:

    results = retrieve(
        test["query"],
        model,
        index,
        mapping,
        top_k=5
    )

    retrieved_ids = [
        result["chunk_id"]
        for result in results
    ]

    if test["expected_chunk"] in retrieved_ids:
        hits += 1
        print(f"PASS: {test['query']}")
    else:
        print(f"MISS: {test['query']}")


recall_at_5 = hits / len(TEST_CASES)

print("\nTotal queries:", len(TEST_CASES))
print("Hits:", hits)
print("Recall@5:", recall_at_5)