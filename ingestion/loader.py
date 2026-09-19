from langchain_community.document_loaders import PyMuPDFLoader

PDF_PATH = "data/raw/medical_rag_knowledge_base.pdf"

def load_pdf():
    loader = PyMuPDFLoader(PDF_PATH)
    documents = loader.load()

    return documents
    

if __name__ == "__main__":
    documents = load_pdf()

    print(f"Total documents/pages loaded: {len(documents)}")

    for document in documents:
        print("\n" + "=" * 80)
        print(f"Page: {document.metadata['page'] + 1}")
        print("=" * 80)
        print(document.page_content)