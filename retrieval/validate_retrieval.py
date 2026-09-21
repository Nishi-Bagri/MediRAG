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


print("===== RETRIEVAL VALIDATION =====")

model = load_model()

index, mapping = load_vectorstore()

passed = 0

for test in TEST_CASES:

    query = test["query"]
    expected_chunk = test["expected_chunk"]

    results = retrieve(
        query,
        model,
        index,
        mapping,
        top_k=5
    )

    retrieved_ids = [
        result["chunk_id"]
        for result in results
    ]

    if expected_chunk in retrieved_ids:
        print(f"PASS: {query}")
        print(f"Expected: {expected_chunk}")
        passed += 1
    else:
        print(f"FAIL: {query}")
        print(f"Expected: {expected_chunk}")
        print("Retrieved:", retrieved_ids)


print("\n===== VALIDATION SUMMARY =====")
print("Total tests:", len(TEST_CASES))
print("Passed:", passed)
print("Failed:", len(TEST_CASES) - passed)

if passed == len(TEST_CASES):
    print("RETRIEVAL VALIDATION: PASS")
else:
    print("RETRIEVAL VALIDATION: FAIL")