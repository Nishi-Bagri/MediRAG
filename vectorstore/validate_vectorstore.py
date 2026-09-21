import json
import faiss


INDEX_FILE = "data/processed/vectorstore/index.faiss"
MAPPING_FILE = "data/processed/vectorstore/chunk_mapping.json"
EMBEDDED_FILE = "data/processed/embedded_chunks.json"


if __name__ == "__main__":

    print("===== FINAL VECTOR DATABASE VALIDATION =====")

    # Load embedded dataset
    with open(EMBEDDED_FILE, "r", encoding="utf-8") as f:
        embedded_data = json.load(f)

    # Load mapping
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Load FAISS index
    index = faiss.read_index(INDEX_FILE)

    expected_count = len(embedded_data)
    expected_dimension = len(embedded_data[0]["embedding"])

    print("Expected vectors:", expected_count)
    print("FAISS vectors:", index.ntotal)

    print("Expected dimension:", expected_dimension)
    print("FAISS dimension:", index.d)

    print("Mapping records:", len(mapping))

    # Validation
    count_pass = index.ntotal == expected_count
    dimension_pass = index.d == expected_dimension
    mapping_pass = len(mapping) == expected_count
    trained_pass = index.is_trained

    print("\n===== VALIDATION RESULTS =====")

    print(
        "Vector count:",
        "PASS" if count_pass else "FAIL"
    )

    print(
        "Vector dimension:",
        "PASS" if dimension_pass else "FAIL"
    )

    print(
        "Mapping count:",
        "PASS" if mapping_pass else "FAIL"
    )

    print(
        "Index trained:",
        "PASS" if trained_pass else "FAIL"
    )

    if (
        count_pass
        and dimension_pass
        and mapping_pass
        and trained_pass
    ):
        print("\nVECTOR DATABASE VALIDATION: PASS")
    else:
        print("\nVECTOR DATABASE VALIDATION: FAIL")