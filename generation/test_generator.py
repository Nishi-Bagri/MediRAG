from generation.generator import (
    create_client, 
    build_context,
    build_prompt,
    generate_answer
)

client = create_client()

retrieved_chunks = [
    {
        "page_content": "HbA1c reflects average blood glucose exposure over several months."
    },
    {
        "page_content": "HbA1c is a commonly used laboratory measure."
    }
]


context = build_context(retrieved_chunks)

query = "What is HbA1c?"

prompt = build_prompt(context, query)

answer = generate_answer(client, prompt)

print("\n===== GENERATED ANSWER ======")
print(answer)