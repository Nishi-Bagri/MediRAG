from rag_pipeline import rag_pipeline

query = "What is the exact HbA1c percentage that confirms diabetes?"

answer = rag_pipeline(query)

print("===== USER QUERY =====")
print(query)

print("\n===== GENERATED ANSWER =====")
print(answer)
