def validate_chunks(chunks, min_chunk_length=100):
    print("===== CHUNK VALIDATION =====")

    print("Total chunks:", len(chunks))

    empty_chunks = [
        i for i, chunk in enumerate(chunks)
        if not chunk.strip()
    ]

    print("Empty chunks:", len(empty_chunks))

    if empty_chunks:
        print("Empty chunk indexes:", empty_chunks)
    else:
        print("Empty chunks: PASS")

    lengths = [len(chunk) for chunk in chunks]

    print("Minimum chunk length:", min(lengths))
    print("Maximum chunk length:", max(lengths)) 

    small_chunks = [
        i for i, chunk in enumerate(chunks)
        if len(chunk.strip()) < min_chunk_length
    ]

    print("Small chunks:", len(small_chunks))

    if small_chunks:
        print("Small chunk indexes:", small_chunks)

        for index in small_chunks:
            print(f"\nSmall chunk {index + 1}:")
            print(chunks[index])
    else:
        print("Small chunks: PASS")

    print("Chunk validation complete.")