import json

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open("data/processed/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

#Generate Embedding for all chunks

embedded_chunks = []

for chunk in chunks:
    embedding = model.encode(chunk["page_content"])

    embedded_chunk = {
        "chunk_id": chunk["chunk_id"],
        "page_content": chunk["page_content"],
        "metadata": chunk["metadata"],
        "embedding": embedding.tolist()
    }

    embedded_chunks.append(embedded_chunk)    


print("Total chunks:", len(embedded_chunks))
print("Embedding dimensions:", len(embedded_chunks[0]["embedding"]))
print("Embeddings generated successfully")

#Save embedded dataset

output_path = "data/processed/embedded_chunks.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(embedded_chunks, file, ensure_ascii=False, indent=4)

print("Embedded dataset saved to:", output_path)