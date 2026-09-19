from loader import load_pdf
from collections import Counter
import re

documents = load_pdf()

print("Total pages:", len(documents))


# Check 1: Character count, word count and ratio

for document in documents:
    page = document.metadata["page"] + 1

    character_count = len(document.page_content)
    word_count = len(document.page_content.split())

    if word_count > 0:
        ratio = character_count / word_count
    else:
        ratio = 0

    print(
        f"Page {page}: "
        f"{character_count} characters, "
        f"{word_count} words, "
        f"ratio: {ratio:.2f}"
    )


# Check 2: First line frequency

first_lines = []

for document in documents:
    lines = document.page_content.splitlines()

    if lines:
        first_lines.append(lines[0].strip())


print("\n===== FIRST LINE FREQUENCY =====")

line_counts = Counter(first_lines)

for line, count in line_counts.most_common():
    print(f"{count} times: {line}")

#Check 3: Repeated lines across the entire document

all_lines = []

for document in documents:
    lines = document.page_content.splitlines()

    for line in lines:
        line = line.strip()

        if line:
            all_lines.append(line)

print("\n===== REPEATED LINES =====")

line_counts = Counter(all_lines)

for line, count in line_counts.most_common():
    if count > 1:
        print(f"{count} times: {line}")

#Check 4: Suspicious characters

print("\n===== SUSPICIOUS CHARACTERS =====")

suspicious_characters = ["�", "□"]

for document in documents:
    page = document.metadata["page"] + 1
    text = document.page_content

    for character in suspicious_characters:
        if character in text:
            print(
                f"Page {page}: "
                f"Suspicious character found: {repr(character)}"
            )

#Check 5: Whitespace problems

print("\n===== WHITESPACE CHECK =====")

for document in documents:
    page = document.metadata["page"] + 1
    text = document.page_content

    # Check for multiple spaces

    if "  " in text:
        print(f"Page {page}: Multiple consecutive spaces found")

    # Check for excessive blank lines
    if "\n\n\n" in text:
        print(f"Page {page}: Excessive blank lines found")


# Check 6: Suspicious word boundaries

print("\n===== SUSPICIOUS WORD BOUNDARIES =====")

for document in documents:
    page = document.metadata["page"] + 1
    text = document.page_content

    # Find words containing 3 or more lowercase letters
    # followed immediately by an uppercase letter.
    matches = re.findall(r"\b[a-z]{3,}[A-Z][a-z]*\b", text)

    if matches:
        print(f"Page {page}: {matches}")

# Check 7: Duplicate pages

print("\n===== DUPLICATE PAGES =====")

page_texts = {}

for document in documents:
    page = document.metadata["page"] + 1
    text = document.page_content.strip()

    if text in page_texts:
        print(
            f"Page {page} is duplicate of "
            f"Page {page_texts[text]}"
        )
    else:
        page_texts[text] = page


# Check 8: Metadata validation

print("\n===== METADATA CHECK =====")

required_metadata = ["source", "page"]

for document in documents:
    page = document.metadata.get("page", "MISSING")
    source = document.metadata.get("source", "MISSING")

    if page == "MISSING" or source == "MISSING":
        print(
            f"Page {page}: "
            f"source={source}, "
            f"page_metadata={page}"
        )

# Final profiler summary

print("\n===== PROFILING COMPLETE =====")
print(f"Total pages analyzed: {len(documents)}")
print("Data profiling checks completed: 11")
print("Potential cleaning candidate: repeated page header")