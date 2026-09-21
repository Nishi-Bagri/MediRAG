import json

MAPPING_FILE = "data/processed/vectorstore/chunk_mapping.json"

if __name__ == "__main__":
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    positions = list(mapping.keys())
    chunk_ids = [item["chunk_id"] for item in mapping.values()]

    valid_fields = all(
       "chunk_id" in item
       and "page_content" in item
       and "metadata" in item
       for item in mapping.values() 
    )

    print("===== CHUNK MAPPING VALIDATION =====")
    print("Total mappings:", len(mapping))
    print("Position count:", len(positions))
    print("Unique chunk IDs:", len(set(chunk_ids)))
    print("Fields valid:", valid_fields)


    if (
        len(mapping) == 66
        and len(set(chunk_ids)) == 66
        and positions == [str(i) for i in range(66)]
        and valid_fields
    ):
        print("MAPPING VALIDATION: PASS")
    else:
        print("MAPPING VALIDATION: FAIL")