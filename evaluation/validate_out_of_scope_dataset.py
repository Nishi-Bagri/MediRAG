import json

DATASET_FILE = "evaluation/out_of_scope_dataset.json"

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print("===== OUT_OF_SCOPE DATASET =====")
print("Total questions:", len(dataset))

for item in dataset:
    print("/nQuestion:", item["question"])

print("\nOut-of-scope dataset loas successfully.")