import json

from rag_pipeline import rag_pipeline

DATASET_FILE = "evaluation/generation_dataset.json"

#Load generation evaluation dataset

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)


print("===== GENERATION EVALUATION =====")

total = 0
passed = 0


for item in dataset:

    question = item["question"]
    expected_type = item["expected_type"]

    answer = rag_pipeline(question)

    answer_lower = answer.lower()

    passed_case = False

    if expected_type == "supported":

        #Supported questions should produce an actual answer
        if (
            "does not contain enough information" not in answer_lower and len(answer.strip()) > 0
        ):
            passed_case = True

    elif expected_type in ["unsupported", "out_of_scope"]:

         # These questions should trigger the knowledge-base limitation
        if (
            "does not contain enough information" in answer_lower
            or "does not contain a specific" in answer_lower
        ):
            passed_case = True

    total += 1

    if passed_case:
        passed += 1


    print("\nQuestion:", question)
    print("Expected type:", expected_type)
    print("Result:", "PASS" if passed_case else "FAIL")

print("\n===== GENERATION METRICS =====")
print("Total questions:", total)
print("Passed:", passed)
print("Failed:", total - passed)

accuracy = passed / total

print(f"Generation evaluation score: {accuracy:.2f}")
print(f"Generation evaluation percentage: {accuracy * 100:.2f}%")