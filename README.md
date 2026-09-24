# MediRAG

A Retrieval-Augmented Generation (RAG) pipeline built from scratch — from a medical PDF to grounded, hallucination-safe LLM answers.

> ⚠️ **Educational project.** Uses a synthetic medical knowledge base. Not a diagnostic tool or a substitute for professional medical advice.

---

## Pipeline

```
PDF → Ingestion → Cleaning → Chunking → Embeddings → FAISS → Retrieval → Prompt → LLM → Evaluation
```

## Tech Stack

Python · LangChain · PyMuPDF · Sentence Transformers · FAISS · NumPy

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)

## Results

| Metric | Value |
|---|---:|
| Source pages | 19 |
| Chunks | 66 |
| Recall@5 | 100% |
| MRR | 0.75 |
| Precision@5 | 20% |
| Generation accuracy | 100% (5/5) |
| Hallucination safety | 100% (5/5) |

The system correctly answers questions grounded in the knowledge base **and** refuses to answer out-of-scope or unsupported questions:

```
Q: What is the capital of France?
A: The provided knowledge base does not contain enough information to answer that question.
```

## Example

```
Q: What is HbA1c?
A: HbA1c is a laboratory measure reflecting average blood glucose
   exposure over a period of roughly the preceding few months.
```

## Project Structure

```
MediRAG/
├── data/                # raw PDF + processed JSON (cleaned, chunks, embeddings)
├── ingestion/            # loading, profiling, cleaning, validation
├── chunking/             # boundary-aware custom chunker
├── embeddings/           # embedding generation + validation
├── vectorstore/          # FAISS index
├── retrieval/            # semantic retriever
├── evaluation/            # recall, MRR, precision, generation, hallucination tests
├── .py        # end-to-end pipeline
└── test_.py
```

## Getting Started

```bash
# Activate environment
.\venv\Scripts\Activate.ps1

# Run the full pipeline
python test_.py
```

### Evaluation

```bash
python -m evaluation.validate_dataset
python -m evaluation.test_retrieval
python -m evaluation.test_mrr
python -m evaluation.test_precision
python -m evaluation.validate_generation_dataset
python -m evaluation.test_generation
python -m evaluation.test_out_of_scope
```

## Status

✅ Core RAG pipeline completed.

Ingestion → Cleaning → Chunking → Embeddings → FAISS → Retrieval → Generation → Evaluation

## Roadmap

Improve retrieval precision
Reranking
Hybrid search
Source citations in answers
Streamlit UI
Conversation history
Monitoring/logging
Expanded evaluation dataset

## Disclaimer

MediRAG uses a **synthetic** knowledge base for RAG/software experimentation only. It must not be used for diagnosis, treatment decisions, emergencies, or as a replacement for a qualified healthcare professional.

## Author

**Nishi Bagri** — [GitHub: MediRAG][GitHub: MediRAG](https://github.com/Nishi-Bagri/MediRAG)