import json
from chunking.chunking import chunk_text, merge_small_chunks
from chunking.validate_chunks import validate_chunks

with open("data/processed/cleaned_documents.json", "r", encoding="utf-8") as file:
    cleaned_documents = json.load(file)

all_chunks = []

print("===== ALL DOCUMENTS CHUNKING =====")
print("Documents:", len(cleaned_documents))

for document in cleaned_documents:
    page_number = document["metadata"]["page"] + 1
    text = document["page_content"]

    chunks = chunk_text(
        text,
        chunk_size=500,
        overlap=50
    )

    chunks = merge_small_chunks(
        chunks,
        min_chunk_length=100,
        max_chunk_length=500
    )

    print(
        f"Page {page_number}: "
        f"{len(chunks)} chunks"
    )

    all_chunks.extend(chunks)

print("\n===== ALL CHUNKS VALIDATION =====")

validate_chunks(all_chunks)

