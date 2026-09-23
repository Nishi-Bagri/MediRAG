from retrieval.retriever import load_model, load_vectorstore, retrieve
from generation.generator import (
    create_client,
    build_context,
    build_prompt,
    generate_answer
)

def rag_pipeline(query):

    #load retrieval components
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

    context = build_context(results)

    prompt = build_prompt(context, query)

    client = create_client()

    answer = generate_answer(client, prompt)

    return answer