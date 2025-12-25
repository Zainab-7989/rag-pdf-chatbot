import os
import gradio as gr
import requests
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# =============================
# CONFIG
# =============================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"  # SAFE FOR HF + FREE GROQ

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =============================
# GLOBAL STATE
# =============================

index = None
chunks = []
sources = []

# =============================
# PDF PROCESSING
# =============================

def load_pdfs(files):
    global index, chunks, sources

    chunks = []
    sources = []

    for file in files:
        reader = PdfReader(file)
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                chunks.append(text.strip())
                sources.append(f"{file.name} - page {page_no}")

    if not chunks:
        return "No readable text found in PDFs."

    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return f"Loaded {len(files)} PDF(s) | Indexed {len(chunks)} chunks"

# =============================
# RAG + GROQ CALL
# =============================

def ask_question(question):
    global index, chunks, sources

    if index is None:
        return "Please upload PDF(s) first.", ""

    q_embedding = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(q_embedding, k=4)

    context = "\n\n".join(chunks[i] for i in I[0])
    used_sources = [sources[i] for i in I[0]]

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say so clearly.

Context:
{context}

Question:
{question}
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful academic assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        data = r.json()

        if "error" in data:
            return f"Groq error: {data['error'].get('message')}", ""

        answer = data["choices"][0]["message"]["content"]
        source_text = "\n".join(f"- {s}" for s in used_sources)

        return answer, source_text

    except Exception as e:
        return f"Request failed: {str(e)}", ""

# =============================
# UI
# =============================

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("##  RAG PDF Chatbot (Gradio + Groq API)")

    pdf_files = gr.File(
        label="Upload PDF(s)",
        file_types=[".pdf"],
        file_count="multiple"
    )

    upload_btn = gr.Button("Process PDFs")
    upload_status = gr.Textbox(label="Upload Info", interactive=False)

    question = gr.Textbox(label="Ask a question")
    answer = gr.Textbox(label="Answer", lines=8)
    source_box = gr.Textbox(label="Sources", lines=4)

    ask_btn = gr.Button("Ask")

    upload_btn.click(
        load_pdfs,
        inputs=pdf_files,
        outputs=upload_status
    )

    ask_btn.click(
        ask_question,
        inputs=question,
        outputs=[answer, source_box]
    )

demo.launch()
