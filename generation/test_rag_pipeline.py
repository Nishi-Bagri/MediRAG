from retrieval.retriever import load_model, load_vectorstore, retrieve

from generation.generator import (
    create_client,
    build_context,
    build_prompt,
    generate_answer
)

query = "What is the capital of France?"

#Load retrieval components
model = load_model()

index, mapping = load_vectorstore()

#Retrieve relevant chunks
results = retrieve(
    query=query,
    model=model,
    index=index,
    mapping=mapping,
    top_k=5
)

#Create Groq client
client = create_client()

#Build context
context = build_context(results)

print("\n===== RETRIEVED CONTEXT =====")
print(context)

#Build prompt
prompt = build_prompt(context, query)

print("\n===== PROMPT =====")
print(prompt)

#Generate answer
answer = generate_answer(client, prompt)

print("====== USER QUERY ======")
print(query)

print("\n===== GENERATED ANSWER =====")
print(answer)

