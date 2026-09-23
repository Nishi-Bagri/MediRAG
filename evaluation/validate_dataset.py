import json

DATASET_FILE = "evaluation/evaluation_dataset.json"

def load_dataset():
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


dataset = load_dataset()

print("===== EVALUATION DATASET =====")
print(f"Total questions: {len(dataset)}")

for item in dataset:
    print(f"\nQuestion: {item['question']}")
    print(f"Relevant chunks: {item['relevant_chunks']}")

print("\Evaluation dataset loaded successfully.")