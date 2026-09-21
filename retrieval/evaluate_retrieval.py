from retriever import load_model, load_vectorstore, retrieve


POSITIVE_CASES = [
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


NEGATIVE_CASES = [
    "What is the capital of France?",
    "How do I repair a car engine?",
    "What is the population of Japan?"
]


print("===== MEDIRAG RETRIEVAL EVALUATION =====")

model = load_model()
index, mapping = load_vectorstore()


# Positive tests

print("\n===== POSITIVE QUERIES =====")

positive_hits = 0

for test in POSITIVE_CASES:

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
        positive_hits += 1
        print("PASS:", test["query"])
    else:
        print("FAIL:", test["query"])


# Negative tests

print("\n===== NEGATIVE QUERIES =====")

for query in NEGATIVE_CASES:

    results = retrieve(
        query,
        model,
        index,
        mapping,
        top_k=5
    )

    print(f"\nQuery: {query}")

    if results:
        print("Candidates returned:", len(results))
        print("Top distance:", results[0]["distance"])
    else:
        print("No results")


print("\n===== SUMMARY =====")

print("Positive queries:", len(POSITIVE_CASES))
print("Positive hits:", positive_hits)
print(
    "Positive Recall@5:",
    positive_hits / len(POSITIVE_CASES)
)

print("Negative queries:", len(NEGATIVE_CASES))