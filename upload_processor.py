import pymupdf

from chunking.chunking import (
    chunk_text,
    merge_small_chunks
)


def process_uploaded_pdf(pdf_bytes, source_name):
    """
    Extract text from an uploaded PDF and create chunks.
    """

    pdf_document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    chunk_records = []

    for page_index, page in enumerate(pdf_document):

        page_number = page_index + 1

        text = page.get_text()

        if not text.strip():
            continue

        chunks = chunk_text(
            text,
            chunk_size=500,
            overlap=50
        )

        chunks = merge_small_chunks(
            chunks,
            min_chunk_length=100,
            max_chunk_length=500
        )

        for chunk_index, chunk in enumerate(chunks):

            chunk_record = {
                "chunk_id": (
                    f"page_{page_number}_chunk_{chunk_index + 1}"
                ),
                "page_content": chunk,
                "metadata": {
                    "page": page_number,
                    "source": source_name
                }
            }

            chunk_records.append(chunk_record)

    pdf_document.close()

    return chunk_records