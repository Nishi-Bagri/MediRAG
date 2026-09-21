def find_sentence_boundary(text, start, end):
    boundaries = [".", "?", "!"]

    positions = []

    for boundary in boundaries:
        position = text.rfind(boundary, start, end)

        if position != -1:
            positions.append(position)

    if positions:
        return max(positions) + 1

    return end 

def find_word_boundary(text, position):
    if position <= 0:
        return position

    while position < len(text) and not text[position].isspace():
        position += 1

    while position < len(text) and text[position].isspace():
        position += 1

    return position

def find_paragraph_boundary(text, start, end):
    position = text.rfind("\n\n", start, end)

    if position != -1:
        return position

    return end

def find_word_end(text, start, end):
    position = text.rfind(" ", start, end)

    if position != -1:
        return position

    return end

def merge_small_chunks(chunks, min_chunk_length = 100, max_chunk_length = 500):
    merged_chunks = []

    for chunk in chunks:
        if (
            merged_chunks
            and len(chunk.strip()) < min_chunk_length
            and len(merged_chunks[-1]) + len(chunk) <= max_chunk_length
        ):

            merged_chunks[-1] += " " + chunk.strip()
        else:
            merged_chunks.append(chunk)

    return merged_chunks

def find_best_boundary(text, start, end):
    paragraph_boundary = find_paragraph_boundary(text, start, end)

    if paragraph_boundary != end:
        return paragraph_boundary

    sentence_boundary = find_sentence_boundary(text, start, end)

    if sentence_boundary != end:
        return sentence_boundary

    word_boundary = find_word_end(text, start, end)

    if word_boundary != end:
        return word_boundary

    return end


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        boundary = find_best_boundary(text, start, end)

        chunk = text[start:boundary]
        chunks.append(chunk)

        if boundary >= len(text):
            break

        next_start = boundary - overlap

        next_start = find_word_boundary(text, next_start)

        if next_start <= start:
            next_start = boundary

        start = next_start

    return chunks