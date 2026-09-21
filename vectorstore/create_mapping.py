import json

INPUT_FILE = "data/processed/embedded_chunks.json"
OUTPUT_FILE = "data/processed/vectorstore/chunk_mapping.json"

def load_chunks():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def create_mapping(data):
    mapping = {}

    for index, item in enumerate(data):
        mapping[str(index)] = {
            "chunk_id": item["chunk_id"],
            "page_content": item["page_content"],
            "metadata": item["metadata"]
        }

    return mapping

if __name__ == "__main__":
    data = load_chunks()

    mapping = create_mapping(data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print("===== CHUNK MAPPING CREATED =====")
    print("Mappings:", len(mapping))
    print("First mapping:", mapping["0"])
    print("Last mapping:", mapping[str(len(mapping) - 1)])