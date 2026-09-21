import json

INPUT_FILE = "data/processed/embedded_chunks.json"

EXPECTED_CHUNK = 66
EXPECTED_DIAMENSIONS = 384

#Load embedded dataset
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    embedded_chunks = json.load(file)

print("===== EMBEDDING DATASET VALIDATION =====")

#Check total records

print("Total records:", len(embedded_chunks))

if len(embedded_chunks) == EXPECTED_CHUNK:
    print("Chunk count: PASS")
else:
    print("Chunk count: FAIL")


invalid_chunks = 0
chunk_ids = set()

# validate every record

for chunk in embedded_chunks:
    required_fields = [
        "chunk_id",
        "page_content",
        "metadata",
        "embedding"
    ]

    if not all(field in chunk for field in required_fields):
        invalid_chunks += 1
        continue

    #check duplicate chunk IDs

    chunk_id = chunk["chunk_id"]

    if chunk_id in chunk_ids:
        invalid_chunks += 1

    chunk_ids.add(chunk_id)

    #check embedding

    embedding = chunk["embedding"]

    if not embedding:
        invalid_chunks += 1
        continue

    if len(embedding) != EXPECTED_DIAMENSIONS:
        invalid_chunks += 1

    #Check metadata

    metadata = chunk["metadata"]

    if "page" not in metadata:
        invalid_chunks += 1

    if "source" not in metadata:
        invalid_chunks += 1

#Final Result

print("Invalid chunks:", invalid_chunks)

if invalid_chunks == 0:
    print("Dataset validation : PASS")
else:
    print("Dataset validation: FAIL")

print("Unique chunk IDs:", len(chunk_ids))