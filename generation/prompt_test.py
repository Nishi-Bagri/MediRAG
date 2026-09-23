import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


context = """
Diabetes is a chronic condition that affects how the body
regulates blood glucose. Common symptoms may include
increased thirst, frequent urination, and fatigue.
"""

question = "What percentage of people with diabetes develop kidney disease?"

prompt =  f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer based only on the provided context.
"""

response = client.chat.completions.create(
    model = "openai/gpt-oss-20b",
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
)
print("===== GENERATED ANSWER =====")
print(response.choices[0].message.content)