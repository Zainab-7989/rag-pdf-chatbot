import os
import gradio as gr
import nltk
import numpy as np
import requests
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

# -------------------- SETUP --------------------

nltk.download("punkt", quiet=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------- GROQ CALL (SAFE) --------------------

def groq_chat(messages):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-70b-versatile",
        "messages": messages,
        "temperature": 0.3
    }

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=25
        )
        r.raise_for_status()
        data = r.json()

        if "choices" not in data or len(data["choices"]) == 0:
            return "Model did not return a response. Please try again."

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error contacting model: {str(e)}"

# -------------------- PDF PROCESSING --------------------

def extract_text(files):
    docs = []
    for f in files:
        reader = PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append({
                    "text": text,
                    "source": f"{f.name} - page {i+1}"
                })
    return docs

def chunk_text(text, size=300, overlap=80):
    if len(text) <= size:
        third = len(text) // 3
        return [text[:third], text[third:2*third], text[2*third:]]

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def build_chunks(docs):
    chunks = []
    for d in docs:
        parts = chunk_text(d["text"])
        for p in parts:
            if p.strip():
                chunks.append({
                    "text": p,
                    "source": d["source"]
                })
    return chunks

def build_index(chunks):
    texts = [c["text"] for c in chunks]
    embeds = EMBEDDER.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeds.shape[1])
    index.add(embeds)
    return index

def retrieve(question, index, chunks, k=4):
    q_emb = EMBEDDER.encode([question], convert_to_numpy=True)
    _, idxs = index.search(q_emb, k)
    return [chunks[i] for i in idxs[0]]

# -------------------- ANSWER --------------------

def answer_question(files, question, history):
    if not files or not question.strip():
        return "Upload PDF(s) and ask a question.", history

    docs = extract_text(files)
    chunks = build_chunks(docs)

    if not chunks:
        return "No readable text found in PDFs.", history

    index = build_index(chunks)
    retrieved = retrieve(question, index, chunks)

    context = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in retrieved
    )

    history_text = ""
    for q, a in history[-3:]:
        history_text += f"Q: {q}\nA: {a}\n\n"

    messages = [
        {
            "role": "system",
            "content": "Answer strictly from the provided context. If not found, say so."
        },
        {
            "role": "user",
            "content": f"""
Conversation history:
{history_text}

Context:
{context}

Question:
{question}
"""
        }
    ]

    answer = groq_chat(messages)

    sources = sorted(set(c["source"] for c in retrieved))
    citation = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

    final = answer + citation
    history.append((question, final))
    return final, history

# -------------------- UI --------------------

with gr.Blocks() as demo:
    gr.Markdown("### Built with Gradio + Groq API")

    pdfs = gr.File(
        file_types=[".pdf"],
        file_count="multiple",
        label="Upload PDF(s)"
    )

    question = gr.Textbox(label="Ask a question")
    answer = gr.Textbox(label="Answer", lines=10)

    state = gr.State([])

    btn = gr.Button("Ask")
    btn.click(answer_question, [pdfs, question, state], [answer, state])

demo.launch()
