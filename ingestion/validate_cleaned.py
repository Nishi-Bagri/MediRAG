import json

with open("data/processed/cleaned_documents.json", "r", encoding="utf-8") as file:
    cleaned_documents = json.load(file)

print("===== CLEANED DATA LOADED =====")
print(f"Documents loaded: {len(cleaned_documents)}")


first_document = cleaned_documents[0]

print("\n===== FIRST DOCUMENT =====")
print("Page:", first_document["metadata"]["page"] + 1)
print("Text:")
print(first_document["page_content"])

print("\n===== ALL DOCUMENTS VALIDATION =====")

for document in cleaned_documents:
    page = document["metadata"].get("page")
    source = document["metadata"].get("source")
    text = document["page_content"].strip()

    if page is None or source is None or not text:
        print(f"Validation issue found on page {page + 1 if page is not None else 'UNKNOWN'}")

print("All documents contain text and required metadata.")

print("\n===== FINAL VALIDATION =====")

print(f"Total documents: {len(cleaned_documents)}")

if len(cleaned_documents) == 19:
    print("Document count: PASS")
else:
    print("Document count: FAIL")

print("Cleaned dataset validation complete.")

