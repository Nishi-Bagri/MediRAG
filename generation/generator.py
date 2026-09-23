import os 
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def create_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found")

    return Groq(api_key=api_key)

def build_context(retrieved_chunks):
    context = "\n\n".join(
        chunk["page_content"]
        for chunk in retrieved_chunks
    )

    return context

def build_prompt(context, query):
    prompt = f"""
You are a medical information assistant.

Answer the user's question using only the provided context.

If the answer is not available in the context, clearly say that the provided knowledge base does not contain enough information to answer the question.

Do not invent or assume information.

Context:
{context}

Question:
{query}

Answer:

"""

    return prompt

def generate_answer(client, prompt):
    response = client.chat.completions.create(
        model ="openai/gpt-oss-20b",
        messages = [
            {
                "role":"user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content