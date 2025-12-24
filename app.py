import gradio as gr
import os
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ================= CONFIG =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing in Hugging Face Secrets")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ============== STATE =====================
faiss_index = None
chunks = []
chat_history = []

# ============== PDF CHUNKING ==============
def chunk_text(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def load_pdf(file):
    global faiss_index, chunks, chat_history
    chat_history = []

    reader = PdfReader(file)
    full_text = ""

    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            full_text += txt + "\n"

    if not full_text.strip():
        return "PDF has no extractable text."

    raw_chunks = chunk_text(full_text)

    chunks = [{"text": c, "source": file.name} for c in raw_chunks]

    embeddings = embedder.encode([c["text"] for c in chunks])
    dim = embeddings.shape[1]

    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(embeddings))

    return f"PDF loaded ({len(chunks)} chunks indexed)"

# ============== RETRIEVAL =================
def retrieve(query, k=5):
    q_emb = embedder.encode([query])
    _, idxs = faiss_index.search(np.array(q_emb), k)
    return [chunks[i] for i in idxs[0]]

# ============== GROQ CALL =================
def call_groq(messages):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
    }

    try:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        data = r.json()

        if "choices" not in data or not data["choices"]:
            return None

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return None

# ============== QA ========================
def ask_question(question):
    if not question.strip():
        return "Please enter a question."

    if faiss_index is None:
        return "Please upload a PDF first."

    retrieved = retrieve(question)

    context = "\n\n".join(c["text"] for c in retrieved)

    history = ""
    for q, a in chat_history[-3:]:
        history += f"Q: {q}\nA: {a}\n\n"

    messages = [
        {
            "role": "system",
            "content": (
                "Answer ONLY from the provided context. "
                "If the answer is not in the document, say so clearly."
            ),
        },
        {
            "role": "user",
            "content": f"{history}\nContext:\n{context}\n\nQuestion:\n{question}",
        },
    ]

    answer = call_groq(messages)

    if answer is None:
        return "Model did not return a response. Please try again."

    sources = sorted(set(c["source"] for c in retrieved))
    citation = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

    chat_history.append((question, answer))
    return answer + citation

# ============== UI ========================
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("### RAG-Based PDF Chatbot (Groq + FAISS)")
    gr.Markdown(
        "Answers are generated strictly from uploaded PDFs with citations.\n\n"
        "**Built with Gradio + Groq API**"
    )

    file = gr.File(label="Upload PDF", file_types=[".pdf"])
    status = gr.Textbox(label="Status", interactive=False)

    question = gr.Textbox(label="Ask a question")
    answer = gr.Textbox(label="Answer", lines=8)

    file.upload(load_pdf, file, status)
    question.submit(ask_question, question, answer)

demo.launch()
