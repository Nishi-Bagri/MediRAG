import json

from retrieval.retriever import (
    load_model,
    load_vectorstore,
    retrieve
)

DATASET_FILE = "evaluation/evaluation_dataset.json"

#load evaluation dataset

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

#Load retrieval components

model = load_model()
index, mapping = load_vectorstore()

print("===== RETRIEVAL EVALUATION =====")

successful = 0
total = 0


for item in dataset:

    question = item["question"]
    expected_chunks = item["relevant_chunks"]

    #Skip question with no expected relevant chunk

    if not expected_chunks:
        continue

    total +=1

    results = retrieve(
        query=question,
        model=model,
        index=index,
        mapping=mapping,
        top_k=5
    )

    retrieved_chunks = [
        result["chunk_id"]
        for result in results
    ]

    found = any(
        chunk_id in retrieved_chunks
        for chunk_id in expected_chunks
    )

    if found:
        successful +=1

    print("\nQuestion:", question)
    print("Expected:", expected_chunks)
    print("Retrieved:", retrieved_chunks)
    print("Relevant chunk found:", found)

#Calculate Recall@5

recall_at_5 = successful / total

print("===== RETRIEVAL METRICS =====")
print("Successful retrievals:", successful)
print("Total evaluable questions:", total)
print(f"Recall@5: {recall_at_5:.2f}")
print(f"Recall@5 percentage: {recall_at_5 * 100:.2f} %")