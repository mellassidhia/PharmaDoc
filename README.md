# PharmaDoc

PharmaDoc is a local Retrieval-Augmented Generation (RAG) system designed for pharmacists. It answers professional questions strictly from pharmaceutical reference documents — clinical guides, drug monographs, therapeutic protocols, and pharmacopeias — without relying on external APIs or internet access.

The system supports French and English and maintains conversational context across multiple exchanges.

---

## Features

- Answers based exclusively on provided PDF documents
- Semantic similarity search with cross-encoder reranking
- Conversational memory (last 3 exchanges)
- Source citation for every response
- Bilingual support: French and English
- Fully offline after initial model download

---

## Architecture

```
User Question
     |
     v
ChromaDB -- similarity search --> top 10 chunks
                                       |
                                  CrossEncoder (reranker)
                                       |
                                  top 4 chunks
                                       |
                      ----------------+
                      |   conversation memory (last 3 turns)
                      |   system prompt
                      v
                  Llama 3.2 via Ollama
                      |
                      v
               Answer + Sources
```

| Component       | Technology                                          |
|----------------|-----------------------------------------------------|
| Document loader | LangChain PyPDFLoader                              |
| Text splitter   | RecursiveCharacterTextSplitter                     |
| Embedding model | paraphrase-multilingual-MiniLM-L12-v2 (HuggingFace)|
| Vector store    | ChromaDB (local)                                   |
| Reranker        | cross-encoder/ms-marco-MiniLM-L-6-v2               |
| Memory          | Python list (last 3 Q&A pairs)                     |
| LLM             | Llama 3.2 via Ollama                               |
| Interface       | Streamlit                                          |

---

## Project Structure

```
pharmadoc/
|
|-- docs/               <- Place pharmaceutical PDF documents here
|-- ingest.py           <- Load, chunk, embed, and store documents
|-- rag.py              <- Retrieve, rerank, and generate answers
|-- app.py              <- Streamlit chat interface
|-- requirements.txt    <- Python dependencies
|-- chroma_db/          <- Generated automatically after ingestion
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/van1shp0tion/PharmaDoc.git
cd pharmadoc
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama and pull the language model

Download Ollama from https://ollama.com then run:

```bash
ollama pull llama3.2
```

---

## Usage

### Step 1 — Add documents

Place your pharmaceutical PDF files (drug monographs, clinical protocols, pharmacology courses) in the `docs/` folder.

### Step 2 — Ingest documents

```bash
python ingest.py
```

This loads, chunks, embeds, and stores all PDFs in the local vector database. Run this once, or again whenever new documents are added.

### Step 3 — Launch the interface

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Example Questions

```
Quelles sont les contre-indications de la metformine ?
Y a-t-il une interaction entre l'amoxicilline et la warfarine ?
Quelle est la posologie du paracetamol chez l'insuffisant renal ?
Summarize the opioid dispensing protocol.
What are the recommended dose adjustments for renal impairment?
```

---

## Constraints

- Runs entirely offline after installation
- No paid API or external service required
- CPU compatible (GPU optional for faster embedding)
- Answers strictly from provided documents — no hallucination

---
## License

This project was developed as part of an academic assignment. It is intended for educational use only.

See the full license in the [LICENSE](LICENSE) file.
