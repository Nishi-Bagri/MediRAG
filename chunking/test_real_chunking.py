import json

from chunking.chunking import chunk_text
from chunking.validate_chunks import validate_chunks
from chunking.chunking import merge_small_chunks

with open(
    "data/processed/cleaned_documents.json",
    "r",
    encoding="utf-8"
) as file:
    cleaned_documents = json.load(file)


first_document = cleaned_documents[0]

page_number = first_document["metadata"]["page"] + 1
text = first_document["page_content"]


chunks = chunk_text(
    text,
    chunk_size=500,
    overlap=50
)

print("\n===== CHUNK DETAILS =====")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}")
    print("Length:", len(chunk))
    print("Start:", repr(chunk[:50]))
    print("End:", repr(chunk[-50:]))


chunks = merge_small_chunks(
    chunks,
    min_chunk_length=100,
    max_chunk_length=500
)


print("===== REAL DOCUMENT CHUNKING =====")
print("Page:", page_number)

validate_chunks(chunks)