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


print("===== FINAL RETRIEVAL VALIDATION =====")

model = load_model()
index, mapping = load_vectorstore()

all_passed = True


# Dimension validation

test_embedding = model.encode(
    ["test query"],
    convert_to_numpy=True
)

embedding_dimension = test_embedding.shape[1]

print("\nEmbedding dimension:", embedding_dimension)
print("FAISS dimension:", index.d)

if embedding_dimension == index.d == 384:
    print("Embedding dimension: PASS")
else:
    print("Embedding dimension: FAIL")
    all_passed = False


# Retrieval validation

for test in TEST_CASES:

    results = retrieve(
        test["query"],
        model,
        index,
        mapping,
        top_k=5
    )

    # Top-K validation

    if len(results) <= 5:
        top_k_pass = True
    else:
        top_k_pass = False
        all_passed = False

    # Distance ordering

    distances = [
        result["distance"]
        for result in results
    ]

    if distances == sorted(distances):
        ordering_pass = True
    else:
        ordering_pass = False
        all_passed = False

    # Expected chunk

    retrieved_ids = [
        result["chunk_id"]
        for result in results
    ]

    expected_pass = test["expected_chunk"] in retrieved_ids

    if not expected_pass:
        all_passed = False

    # Structure

    structure_pass = True

    for result in results:

        if not result["chunk_id"]:
            structure_pass = False

        if not result["page_content"]:
            structure_pass = False

        if not isinstance(result["metadata"], dict):
            structure_pass = False

        if "page" not in result["metadata"]:
            structure_pass = False

        if not isinstance(result["distance"], float):
            structure_pass = False

    if not structure_pass:
        all_passed = False

    print(f"\nQuery: {test['query']}")
    print("Expected chunk:", test["expected_chunk"])
    print("Expected chunk retrieved:", "PASS" if expected_pass else "FAIL")
    print("Top-K validation:", "PASS" if top_k_pass else "FAIL")
    print("Distance ordering:", "PASS" if ordering_pass else "FAIL")
    print("Result structure:", "PASS" if structure_pass else "FAIL")


print("\n===== FINAL RESULT =====")

if all_passed:
    print("RETRIEVAL MODULE VALIDATION: PASS")
else:
    print("RETRIEVAL MODULE VALIDATION: FAIL")