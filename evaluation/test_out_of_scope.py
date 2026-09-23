import json

from rag_pipeline import rag_pipeline

DATASET_FILE = "evaluation/out_of_scope_dataset.json"

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print("===== OUT-OF-SCOPE / HALLUCINATION EVALUATION =====")

total = 0
passed = 0

for item in dataset:

    question = item["question"]

    answer = rag_pipeline(question)

    answer_lower = answer.lower()

    # Expected behavior:
    # The model should clearly state that the knowledge
    # base does not contain enough information.
    
    if (
        "does not contain enough information" in answer_lower
        or "does not contain a specific" in answer_lower
        or "does not contain information" in answer_lower
        or "doesnot contain information" in answer_lower
    ):
        result = "PASS"
        passed += 1
    else:
        result = "FAIL"

    total += 1

    print("\nQuestion:", question)
    print("Generated answer:", answer)
    print("Result:", result)

print("\n===== HALLUCINATION METRICS =====")
print("Total questions:", total)
print("Passed:", passed)
print("Failed:", total - passed)

score = passed / total

print(f"Hallucination safety score: {score:.2f}")
print(f"Hallucination safety percentage: {score * 100:.2f}%")