from retrieval.retriever import load_model, load_vectorstore, retrieve

query = "What is HbA1c?"

model = load_model()
index, mapping = load_vectorstore()

results = retrieve(
    query=query,
    model=model,
    index=index,
    mapping=mapping,
    top_k=5
)

context = "\n\n".join(
    result["page_content"]
    for result in results
)

print("===== RETRIEVED CONTEXT =====")
print(context)