from retriever import load_model, load_vectorstore, retrieve


print("===== MEDIRAG RETRIEVER TEST =====")

model = load_model()

index, mapping = load_vectorstore()

query = "What is HbA1c?"

results = retrieve(
    query,
    model,
    index,
    mapping,
    top_k=5
)

print("\nQuery:", query)

if not results:
    print("No results found.")
else:
    for result in results:
        print(f"\nRank: {result['rank']}")
        print("Chunk ID:", result["chunk_id"])
        print("Distance:", result["distance"])
        print("Page:", result["metadata"]["page"])
        print("Content:", result["page_content"][:200])