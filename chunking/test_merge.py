from chunking.chunking import merge_small_chunks


chunks = [
    "A" * 480,
    "B" * 50
]

merged = merge_small_chunks(
    chunks,
    min_chunk_length=100,
    max_chunk_length=500
)

print("Number of chunks:", len(merged))

for i, chunk in enumerate(merged):
    print(f"Chunk {i + 1} length:", len(chunk))