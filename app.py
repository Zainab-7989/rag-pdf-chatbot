import os
import gradio as gr
import numpy as np
import nltk
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# =========================
# NLTK AUTO-SETUP (FIXED)
# =========================
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

# =========================
# GROQ CLIENT
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable not set")

groq_client = Groq(api_key=GROQ_API_KEY)

# =========================
# EMBEDDING MODEL
# =========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_text_from_pdfs(files):
    documents = []
    for file in files:
        reader = PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append({
                    "text": text,
                    "source": f"{file.name} - page {i+1}"
                })
    return documents

# =========================
# NLTK CHUNKING
# =========================
def chunk_documents(documents, max_words=200):
    chunks = []
    for doc in documents:
        sentences = sent_tokenize(doc["text"])
        current_chunk = []
        word_count = 0

        for sentence in sentences:
            words = sentence.split()
            if word_count + len(words) <= max_words:
                current_chunk.append(sentence)
                word_count += len(words)
            else:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "source": doc["source"]
                })
                current_chunk = [sentence]
                word_count = len(words)

        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "source": doc["source"]
            })

    return chunks

# =========================
# FAISS VECTOR STORE
# =========================
def build_faiss_index(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks

# =========================
# RETRIEVAL
# =========================
def retrieve_chunks(question, index, chunks, k=4):
    q_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_embedding, k)

    retrieved = []
    for idx in indices[0]:
        retrieved.append(chunks[idx])

    return retrieved

# =========================
# ANSWER GENERATION (WITH MEMORY + CITATIONS)
# =========================
def generate_answer(question, retrieved_chunks, history):
    context = "\n\n".join(
        [f"[{c['source']}]\n{c['text']}" for c in retrieved_chunks]
    )

    history_text = ""
    for q, a in history[-3:]:  # last 3 turns
        history_text += f"Q: {q}\nA: {a}\n\n"

    prompt = f"""
You are a document-based assistant.
Answer ONLY using the provided context.
If the answer is not present, say:
"Answer not found in the uploaded documents."

Conversation History:
{history_text}

Context:
{context}

Question:
{question}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    sources = list(set(c["source"] for c in retrieved_chunks))
    citations = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

    return answer + citations

# =========================
# MAIN RAG FUNCTION
# =========================
def rag_chat(pdf_files, question, history):
    if not pdf_files:
        return "Please upload at least one PDF.", history

    documents = extract_text_from_pdfs(pdf_files)
    chunks = chunk_documents(documents)
    index, chunks = build_faiss_index(chunks)
    retrieved = retrieve_chunks(question, index, chunks)

    answer = generate_answer(question, retrieved, history)
    history.append((question, answer))

    return answer, history

# =========================
# GRADIO UI
# =========================
with gr.Blocks(title="RAG PDF Chatbot") as demo:
    gr.Markdown("##  RAG-Based PDF Chatbot (Groq + FAISS)")
    gr.Markdown(
        "Ask questions **only from the uploaded PDFs**. "
        "The chatbot will cite exact pages used in the answer."
    )

    pdf_input = gr.File(
        file_types=[".pdf"],
        file_count="multiple",
        label="Upload PDF files"
    )

    question_input = gr.Textbox(
        label="Ask a question from the documents"
    )

    answer_output = gr.Textbox(
        label="Answer",
        lines=10
    )

    chat_state = gr.State([])

    ask_btn = gr.Button("Ask")

    ask_btn.click(
        fn=rag_chat,
        inputs=[pdf_input, question_input, chat_state],
        outputs=[answer_output, chat_state]
    )

demo.launch()
