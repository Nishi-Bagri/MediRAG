import json

DATASET_FILE = "evaluation/generation_dataset.json"

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print("===== GENERATION EVALUATION DATASET =====")
print("Total questions:", len(dataset))


for item in dataset:
    print("\nQuestion:", item["question"])
    print("Expected type:", item["expected_type"])

print("\nGeneration evaluation dataset loaded successfully.")

