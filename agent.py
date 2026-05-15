from pathlib import Path
import re
import ollama
from rag import retrieve, rerank, build_context, get_source_labels

OLLAMA_MODEL = "llama3.2"
MEMORY_TURNS = 3

ROUTER_PROMPT = """You are a routing assistant. Given a user question, decide which tool to use.

Available tools:
- search_docs: Use when the question is about a specific drug, medication, interaction, contraindication, dosage, protocol, or any clinical/pharmaceutical topic that requires looking up documents.
- calculate_dose: Use when the question asks to calculate a dose based on patient weight or age.
- direct_answer: Use when the question is general knowledge that does not require document lookup (e.g. what is a RAG system, what is pharmacology).

Respond with ONLY one word: search_docs, calculate_dose, or direct_answer."""

PHARMA_PROMPT = """You are PharmaDoc, a professional assistant for pharmacists.
Answer questions strictly based on the pharmaceutical documents provided.

Rules:
- Answer ONLY from the context below. Do not use external knowledge.
- If the answer is not found in the context, say: "This information is not available in the provided documents."
- Be precise and concise. Use bullet points or numbered lists when appropriate.
- Always cite the source document and page number for each piece of information.
- Answer in the same language the user writes in (French or English).
- Never invent or assume information not present in the source documents.
"""

DIRECT_PROMPT = """You are PharmaDoc, a professional assistant for pharmacists.
Answer the question using your general pharmaceutical knowledge.
Be concise and professional.
Answer in the same language the user writes in (French or English)."""


def route(question):
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user",   "content": question},
        ],
    )
    tool = response["message"]["content"].strip().lower()
    if tool not in {"search_docs", "calculate_dose", "direct_answer"}:
        tool = "search_docs"
    return tool


def tool_search_docs(question, memory):
    raw_chunks = retrieve(question)
    top_chunks = rerank(question, raw_chunks)
    context    = build_context(top_chunks)

    messages = [{"role": "system", "content": PHARMA_PROMPT}]
    messages.append({"role": "system", "content": "PHARMACEUTICAL DOCUMENT CONTEXT:\n\n" + context})
    for turn in memory[-MEMORY_TURNS:]:
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": question})

    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    return {
        "answer":  response["message"]["content"],
        "sources": get_source_labels(top_chunks),
        "tool":    "search_docs",
    }


def tool_calculate_dose(question):
    weight = None
    age    = None

    weight_match = re.search(r"(\d+(?:\.\d+)?)\s*kg", question, re.IGNORECASE)
    age_match    = re.search(r"(\d+)\s*(?:ans|years|yo|year)", question, re.IGNORECASE)

    if weight_match:
        weight = float(weight_match.group(1))
    if age_match:
        age = int(age_match.group(1))

    calc_prompt = f"""You are a pharmaceutical dose calculator assistant.
The user asked: {question}
{"Patient weight: " + str(weight) + " kg" if weight else ""}
{"Patient age: " + str(age) + " years" if age else ""}

Using standard pharmaceutical dosing guidelines, calculate the appropriate dose.
Show the formula used and the result clearly.
If weight or age is missing, ask the user to provide it.
Answer in the same language the user writes in (French or English)."""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": calc_prompt}],
    )
    return {
        "answer":  response["message"]["content"],
        "sources": [],
        "tool":    "calculate_dose",
    }


def tool_direct_answer(question, memory):
    messages = [{"role": "system", "content": DIRECT_PROMPT}]
    for turn in memory[-MEMORY_TURNS:]:
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": question})

    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    return {
        "answer":  response["message"]["content"],
        "sources": [],
        "tool":    "direct_answer",
    }


def run(question, memory):
    tool = route(question)

    if tool == "search_docs":
        return tool_search_docs(question, memory)
    elif tool == "calculate_dose":
        return tool_calculate_dose(question)
    else:
        return tool_direct_answer(question, memory)


if __name__ == "__main__":
    memory = []
    print("PharmaDoc Agent - (type 'quit' to exit)")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        result = run(question, memory)
        print(f"\n[Tool used: {result['tool']}]")
        print(f"\nAssistant: {result['answer']}")
        if result["sources"]:
            print("\nSources:")
            for src in result["sources"]:
                print(f"  - {src}")
        memory.append({"question": question, "answer": result["answer"]})
