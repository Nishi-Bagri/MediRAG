from retriever import load_model, load_vectorstore, retrieve


print("===== RETRIEVAL RESULT VALIDATION =====")

model = load_model()
index, mapping = load_vectorstore()

query = "What is diabetes?"

results = retrieve(
    query,
    model,
    index,
    mapping,
    top_k=5
)

print("Results returned:", len(results))

valid = True

for result in results:

    required_keys = [
        "rank",
        "chunk_id",
        "page_content",
        "metadata",
        "distance"
    ]

    for key in required_keys:
        if key not in result:
            print(f"FAIL: Missing key '{key}'")
            valid = False

    if not isinstance(result["rank"], int):
        print("FAIL: Rank is not integer")
        valid = False

    if not isinstance(result["chunk_id"], str):
        print("FAIL: Chunk ID is not string")
        valid = False

    if not isinstance(result["page_content"], str):
        print("FAIL: Page content is not string")
        valid = False

    if not isinstance(result["metadata"], dict):
        print("FAIL: Metadata is not dictionary")
        valid = False

    if not isinstance(result["distance"], float):
        print("FAIL: Distance is not float")
        valid = False


if valid:
    print("All result structures are valid.")
    print("RETRIEVAL RESULT VALIDATION: PASS")
else:
    print("RETRIEVAL RESULT VALIDATION: FAIL")