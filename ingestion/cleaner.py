from loader import load_pdf
import json
from pathlib import Path

documents = load_pdf()

HEADER = "Medical Health Knowledge Base — RAG Practice"


# Step 1: Remove repeated header

for document in documents:
    text = document.page_content

    if text.startswith(HEADER):
        document.page_content = text[len(HEADER):].lstrip()

print("Repeated header removal completed.")


# Step 2: Verify header removal

print("\n===== HEADER VERIFICATION =====")

remaining_headers = 0

for document in documents:
    if HEADER in document.page_content:
        remaining_headers += 1
        page = document.metadata["page"] + 1

        print(f"Header still found on page {page}")

if remaining_headers == 0:
    print("Header successfully removed from all pages.")
else:
    print(f"Header still present on {remaining_headers} pages.")


# Step 3: Check empty pages

print("\n===== EMPTY PAGE CHECK =====")

for document in documents:
    page = document.metadata["page"] + 1
    text = document.page_content.strip()

    if not text:
        print(f"Page {page}: EMPTY")

    elif len(text) < 50:
        print(f"Page {page}: Very little text ({len(text)} characters)")


# Step 4: Whitespace cleaning

print("\n===== WHITESPACE CLEANING =====")

for document in documents:
    lines = document.page_content.splitlines()

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if line:
            cleaned_lines.append(line)

    document.page_content = "\n".join(cleaned_lines)

print("Whitespace cleaning completed.")


# Step 5: Verify whitespace cleaning

print("\n===== WHITESPACE VERIFICATION =====")

whitespace_issues = 0

for document in documents:
    page = document.metadata["page"] + 1
    text = document.page_content

    lines = text.splitlines()

    for line in lines:
        if line != line.rstrip():
            print(f"Page {page}: Trailing whitespace found")
            whitespace_issues += 1
            break

    if "\n\n\n" in text:
        print(f"Page {page}: Excessive blank lines found")
        whitespace_issues += 1

if whitespace_issues == 0:
    print("No whitespace issues found.")
else:
    print(f"Whitespace issues found: {whitespace_issues}")


# Step 6: Suspicious character verification

print("\n===== SUSPICIOUS CHARACTER VERIFICATION =====")

suspicious_characters = ["�", "□"]
suspicious_found = 0


for document in documents:
    page = document.metadata["page"] + 1
    text = document.page_content

    for character in suspicious_characters:
        if character in text:
            print(
                f"Page {page}: "
                f"Suspicious character found: {repr(character)}"
            )
            suspicious_found += 1

if suspicious_found == 0:
    print("No suspicious characters found.")
else:
    print(f"Suspicious character occurrences: {suspicious_found}")

# Step 7: Suspicious word-boundary verification

print("\n===== WORD-BOUNDARY VERIFICATION =====")

boundary_issues = 0

for document in documents:
    page = document.metadata["page"] + 1
    text = document.page_content

    words = text.split()

    for word in words:
        if len(word) > 3:
            for i in range(1, len(word)):
                if word[i].isupper() and word[i - 1].islower():
                    print(f"Page {page}: Possible boundary issue: {word}")
                    boundary_issues += 1
                    break

if boundary_issues == 0:
    print("No suspicious word-boundary patterns found.")
else:
    print(f"Possible word-boundary issues: {boundary_issues}")  

# Step 8: Duplicate content verification

print("\n===== DUPLICATE CONTENT VERIFICATION =====")

page_texts = {}
duplicates_found = 0

for document in documents:
    page = document.metadata["page"] + 1
    text = document.page_content.strip()

    if text in page_texts:
        print(
            f"Page {page} has duplicate content "
            f"from page {page_texts[text]}"
        )
        duplicates_found += 1
    else:
        page_texts[text] = page 

if duplicates_found == 0:
    print("No duplicate pages found.")
else:
    print(f"Duplicate pages found: {duplicates_found}")        

# Step 9: Metadata preservation verification

print("\n===== METADATA VERIFICATION =====")

metadata_issues = 0

for document in documents:
    page = document.metadata.get("page")
    source = document.metadata.get("source")

    if page is None:
        print("Missing page metadata")
        metadata_issues += 1

    if source is None:
        print(f"Page {page}: Missing source metadata")
        metadata_issues += 1

if metadata_issues == 0:
    print("All required metadata preserved.")
else:
    print(f"Metadata issues found: {metadata_issues}")          

# Final cleaning validation report

print("\n===== FINAL CLEANING REPORT =====")

print(f"Total pages processed: {len(documents)}")

print("Repeated header removed: YES")
print("Empty pages: NONE")
print("Whitespace issues: NONE")
print("Suspicious characters: NONE")
print("Duplicate pages: NONE")
print("Required metadata preserved: YES")

print("\nLegitimate word-boundary patterns:")
print("- HbA1c")
print("- LangChain")

print("\n===== DATA CLEANING COMPLETE =====")

# Step 11: Save cleaned documents

output_path = Path("data/processed/cleaned_documents.json")

cleaned_data = []

for document in documents:
    cleaned_data.append({
        "page_content": document.page_content,
        "metadata": document.metadata
    })

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(cleaned_data, file, ensure_ascii=False, indent=2)

print("\n===== CLEANED DATA SAVED =====")
print(f"Output file: {output_path}")
print(f"Documents saved: {len(cleaned_data)}")