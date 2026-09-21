import json

from chunking.chunking import chunk_text, merge_small_chunks


with open(
    "data/processed/cleaned_documents.json",
    "r",
    encoding="utf-8"
) as file:
    cleaned_documents = json.load(file)


print("===== CHUNKER DATA LOADED =====")
print("Documents:", len(cleaned_documents))


all_chunk_records = []

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

    for i, chunk in enumerate(chunks):

        chunk_record = {
            "chunk_id": f"page_{page_number}_chunk_{i + 1}",
            "page_content": chunk,
            "metadata": {
                "page": page_number,
                "source": document["metadata"]["source"]
            }
        }

        all_chunk_records.append(chunk_record)


print("\n===== FINAL CHUNK DATASET =====")
print("Documents processed:", len(cleaned_documents))
print("Total chunks:", len(all_chunk_records))

print("\n===== FIRST CHUNK =====")
print(all_chunk_records[0])

with open(
    "data/processed/chunks.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        all_chunk_records,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\n===== CHUNKS SAVED =====")
print("File: data/processed/chunks.json")