import json


with open(
    "data/processed/chunks.json",
    "r",
    encoding="utf-8"
) as file:
    chunks = json.load(file)


print("===== CHUNK DATASET VALIDATION =====")

print("Total chunks:", len(chunks))


invalid_chunks = []

for chunk in chunks:

    if not chunk.get("chunk_id"):
        invalid_chunks.append(chunk)

    elif not chunk.get("page_content", "").strip():
        invalid_chunks.append(chunk)

    elif not chunk.get("metadata", {}).get("page"):
        invalid_chunks.append(chunk)

    elif not chunk.get("metadata", {}).get("source"):
        invalid_chunks.append(chunk)


print("Invalid chunks:", len(invalid_chunks))

if invalid_chunks:
    print("Dataset validation: FAIL")
else:
    print("Dataset validation: PASS")


pages = sorted(
    set(chunk["metadata"]["page"] for chunk in chunks)
)

print("Pages represented:", len(pages))
print("Page range:", pages[0], "to", pages[-1])


print("\n===== VALIDATION COMPLETE =====")