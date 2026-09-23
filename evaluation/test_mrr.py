import json

from retrieval.retriever import (
    load_model,
    load_vectorstore,
    retrieve
)

DATASET_FILE = "evaluation/evaluation_dataset.json"

#Load evaluation dataset

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

#Load retrieval Components

model = load_model()
index, mapping = load_vectorstore()

TOP_K =5

reciprocal_ranks = []

print("===== MRR EVALUATION =====")

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

    reciprocal_rank = 0

    for rank, result in enumerate(results, start=1):

        chunk_id = result["chunk_id"]

        if chunk_id in expected_chunks:
            reciprocal_rank = 1 / rank
            break


    reciprocal_ranks.append(reciprocal_rank)

    print("\nQuestion:", question)
    print("Expected:", expected_chunks)

    if reciprocal_rank > 0:
            print("Reciprocal Rank:", f"{reciprocal_rank:.2f}")
    else:
            print("Relevant chunk not found in Top-5")

#Calculate MRR

mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

print("\n===== MRR METRIC =====")
print("Total evaluable questions:", len(reciprocal_ranks))
print(f"MRR: {mrr:.2f}")