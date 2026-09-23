import json

from retrieval.retriever import (
    load_model,
    load_vectorstore,
    retrieve
    )

DATA_FILE = "evaluation/evaluation_dataset.json"

#Load evaluation dataset

with open(DATA_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

#Load retrieval components

model = load_model()
index, mapping = load_vectorstore()

TOP_K = 5

total_precision = 0
total_questions = 0

print("===== PRECISION@5 EVALUATION =====")

for item in dataset:

    question = item["question"]
    expected_chunks = item["relevant_chunks"]

    #Skip out-of-scope questions
    if not expected_chunks:
        continue

    results = retrieve(
        query=question,
        model=model,
        index=index,
        mapping=mapping,
        top_k=TOP_K
    )

    retrieved_chunks = [
        result["chunk_id"]
        for result in results
    ]

    relevant_retrieved = sum(
        1
        for chunk_id in retrieved_chunks
        if chunk_id in expected_chunks
    )

    precision = relevant_retrieved / TOP_K

    total_precision += precision
    total_questions += 1

    print("\nQuestion:", question)
    print("Expected:", expected_chunks)
    print("Retrieved:", retrieved_chunks)
    print("Relevant retrieved:", relevant_retrieved)
    print(f"Precision@5:{precision:.2f}")

#Calculate mean Precision@5

mean_precision = total_precision / total_questions

print("\n===== PRECISION@5 METRIC =====")
print("Total evaluable questions:", total_questions)
print(f"Mean Precision@5: {mean_precision:.2f}")
print(f"Precision@5 percentage: {mean_precision * 100:.2f}%")