retrieved_chunks = [
    {
        "chunk_id": "page_7_chunk_3",
        "page": 7,
        "content": """laboratory criteria rather than symptoms alone. Common tests
include measures of blood glucose and glycated hemoglobin (HbA1c), with
interpretation depending on the clinical context."""
    },
    {
        "chunk_id": "page_8_chunk_1",
        "page": 8,
        "content": """Diabetes: Key Concepts for Retrieval.
Blood glucose is the amount of glucose circulating in the blood."""
    }
]

context = "\n\n".join(
    chunk["content"]
    for chunk in retrieved_chunks
)

print("===== CONTEXT =====")
print(context)