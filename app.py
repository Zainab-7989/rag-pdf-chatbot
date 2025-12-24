import gradio as gr
import os
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import nltk

nltk.download("punkt")

# ================= CONFIG =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in Hugging Face Secrets")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ============== GLOBAL STATE ==============
faiss_index = None
chunks = []
chat_history = []

# ============== PDF PROCESSING ============
def load_pdf(file):
    global faiss_index, chunks, chat_history
    chat_history = []

    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    sentences = nltk.sent_tokenize(text)
    chunks = [{"text": s, "source": file.name} for s in sentences]

    embeddings = embedder.encode([c["text"] for c in chunks])
    dim = embeddings.shape[1]

    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(embeddings))

    return f"Loaded {len(chunks)} chunks from PDF."

# ============== RETRIEVAL =================
def retrieve(query, k=5):
    q_emb = embedder.encode([query])
    _, idxs = faiss_index.search(np.array(q_emb), k)
    return [chunks[i] for i in idxs[0]]

# ============== GROQ CALL =================
def groq_chat(messages):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]

# ============== ANSWER ====================
def ask_question(question):
    if faiss_index is None:
        return "Please upload a PDF first."

    retrieved = retrieve(question)

    context = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in retrieved
    )

    history_text = ""
    for q, a in chat_history[-3:]:
        history_text += f"Q: {q}\nA: {a}\n\n"

    messages = [
        {
            "role": "system",
            "content": (
                "Answer strictly from the provided context. "
                "If not found, say: Answer not found in the document."
            ),
        },
        {
            "role": "user",
            "content": f"{history_text}\nContext:\n{context}\n\nQuestion:\n{question}",
        },
    ]

    answer = groq_chat(messages)

    sources = sorted(set(c["source"] for c in retrieved))
    citations = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

    chat_history.append((question, answer))
    return answer + citations

# ============== UI ========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## RAG-Based PDF Chatbot (Groq + FAISS)")
    gr.Markdown("Answers are generated strictly from uploaded PDFs with citations.")

    file = gr.File(label="Upload PDF", file_types=[".pdf"])
    status = gr.Textbox(label="Status", interactive=False)

    question = gr.Textbox(label="Ask a question")
    answer = gr.Textbox(label="Answer", lines=10)

    file.upload(load_pdf, file, status)
    question.submit(ask_question, question, answer)

demo.launch()
