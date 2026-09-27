from upload_processor import process_uploaded_pdf
from upload_embeddings import embed_uploaded_chunks
from upload_vectorstore import create_upload_vectorstore

from retrieval.retriever import (
    load_model,
    retrieve
)

from generation.generator import (
    create_client,
    build_context,
    build_prompt,
    generate_answer
)

# Load Test PDF

with open(
    "data/raw/medical_rag_knowledge_base.pdf",
    "rb"
) as file:

    pdf_bytes = file.read()

#Process PDF

chunks = process_uploaded_pdf(
    pdf_bytes,
    "test.pdf"
)

print("Chunks:", len(chunks))

#Embeddings

embedded_chunks = embed_uploaded_chunks(
    chunks
)

print(
    "Embeddings:", len(embedded_chunks)
)

#Temporary FAISS

index, mapping = create_upload_vectorstore(
    embedded_chunks
)

print("FAISS vectors:", index.ntotal)

#RETRIEVAL

model = load_model()

query = "What is HbA1c?"

results = retrieve(
    query=query,
    model=model,
    index=index,
    mapping=mapping,
    top_k=5,
    max_distance=1.5
)

print(
    "Retrieved chunks:", len(results)
)

#Context

context = build_context(
    results
)

#Prompt

prompt = build_prompt(
    context,
    query
)

#Generation

client = create_client()

answer = generate_answer(
    client,
    prompt
)

#Result

print("\n===== Answer =====")
print(answer)

print("\n===== SOURCE =====")

for result in results:
    print(
        result["chunk_id"],
        "| Page:",
        result["metadata"].get("page"),
        "|Distance:",
        result["distance"]
    )