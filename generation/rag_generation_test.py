import os
from dotenv import load_dotenv
from groq import Groq

from retrieval.retriever import load_model, load_vectorstore, retrieve

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

query = "What is the capital of France?"
#Load retrieval components

model = load_model()
index, mapping = load_vectorstore()

#Retrieve relevant chunks

results = retrieve(
    query=query,
    model=model,
    index=index,
    mapping=mapping,
    top_k=5
)

#Build context from retrieved chunks

context = "\n\n".join(
    result["page_content"]
    for result in results
)

#Build prompt
prompt = f"""
You are a medical information assistant.

Answer the user's question using only the provided context.

if the answer is not available in the context, clearly say that the provided knowledge base does not contain enough information to answer the question.

do not invent or assume information.

Context:
{context}

Question:
{query}

Answer:

"""

#Generate Answer
response = client.chat.completions.create(
    model = "openai/gpt-oss-20b",
    messages = [
        {
            "role": "user",
            "content":prompt
        }
    ]
)

print("===== USER QUERY =====")
print(query)

print("\n===== GENERATED ANSWER =====")
print(response.choices[0].message.content)

