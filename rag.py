from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
import ollama

CHROMA_DIR    = "chroma_db"
EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL  = "llama3.2"
TOP_K_RETRIEVE = 10
TOP_K_RERANK   = 4
MEMORY_TURNS   = 3

SYSTEM_PROMPT = """You are PharmaDoc, a professional assistant for pharmacists.
Answer questions strictly based on the pharmaceutical documents provided.

Rules:
- Answer ONLY from the context below. Do not use external knowledge.
- If the answer is not found in the context, say: "This information is not available in the provided documents."
- Be precise and concise. Use bullet points or numbered lists when appropriate.
- Always cite the source document and page number for each piece of information.
- Answer in the same language the user writes in (French or English).
- Never invent or assume information not present in the source documents.
"""

_embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

_vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=_embeddings,
)

_reranker = CrossEncoder(RERANK_MODEL)


def retrieve(query, k=TOP_K_RETRIEVE):
    return _vectorstore.similarity_search(query, k=k)


def rerank(query, chunks, top_n=TOP_K_RERANK):
    if not chunks:
        return []
    pairs  = [(query, chunk.page_content) for chunk in chunks]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_n]]


def build_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        page   = chunk.metadata.get("page", "?")
        parts.append(
            f"[Source {i}: {Path(source).name}, page {page}]\n"
            f"{chunk.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def build_messages(context, memory, question):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({
        "role": "system",
        "content": "PHARMACEUTICAL DOCUMENT CONTEXT:\n\n" + context,
    })
    for turn in memory[-MEMORY_TURNS:]:
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": question})
    return messages


def get_source_labels(chunks):
    seen, labels = set(), []
    for chunk in chunks:
        source = Path(chunk.metadata.get("source", "unknown")).name
        page   = chunk.metadata.get("page", "?")
        label  = f"{source} (page {page})"
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def answer(question, memory):
    raw_chunks = retrieve(question)
    top_chunks = rerank(question, raw_chunks)
    context    = build_context(top_chunks)
    messages   = build_messages(context, memory, question)
    response   = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    reply      = response["message"]["content"]
    return {
        "answer":  reply,
        "sources": get_source_labels(top_chunks),
        "chunks":  top_chunks,
    }


if __name__ == "__main__":
    memory = []
    print("PharmaDoc - RAG Chat (type 'quit' to exit)")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        result = answer(question, memory)
        print(f"\nAssistant: {result['answer']}")
        if result["sources"]:
            print("\nSources:")
            for src in result["sources"]:
                print(f"  - {src}")
        memory.append({"question": question, "answer": result["answer"]})