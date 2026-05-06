import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR      = "docs"
CHROMA_DIR    = "chroma_db"
EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80


def load_pdfs(docs_dir):
    pdf_paths = list(Path(docs_dir).rglob("*.pdf"))
    if not pdf_paths:
        print(f"No PDF files found in '{docs_dir}'.")
        sys.exit(1)
    all_docs = []
    for pdf_path in pdf_paths:
        print(f"Loading: {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        for page in pages:
            page.metadata["source"] = str(pdf_path)
        all_docs.extend(pages)
    print(f"Loaded {len(all_docs)} pages from {len(pdf_paths)} file(s).")
    return all_docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks):
    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Embedding and storing chunks ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectorstore.persist()
    print(f"Vector database saved to '{CHROMA_DIR}'.")


def main():
    print("PharmaDoc - Ingestion Pipeline")
    docs   = load_pdfs(DOCS_DIR)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    print("Ingestion complete. Run: streamlit run app.py")


if __name__ == "__main__":
    main()