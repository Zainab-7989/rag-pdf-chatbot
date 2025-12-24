# RAG-based PDF Question Answering Chatbot

## Project Overview
This project is a Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload PDF documents and ask natural language questions about their content. The chatbot retrieves relevant information from the uploaded document and generates accurate, context-aware answers grounded strictly in the document data.

Unlike general-purpose chatbots, this system does not hallucinate answers. Every response is generated based on retrieved document chunks and includes page-level source citations.

---

## Objectives
- Enable intelligent question answering over PDF documents
- Demonstrate practical implementation of RAG architecture
- Combine NLP preprocessing, vector similarity search, and LLM-based reasoning
- Provide transparent answers with document citations
- Maintain conversational context through chat history

---

## Key Features
- PDF upload and text extraction
- NLTK-based text chunking for better semantic retrieval
- Sentence-transformer embeddings for document representation
- FAISS vector database for efficient similarity search
- Retrieval-Augmented Generation (RAG) pipeline
- Page-level source citations for every answer
- Conversational memory (chat history)
- Clean and simple Gradio-based user interface

---

## System Architecture
1. **Document Ingestion**
   - User uploads a PDF document
   - Text is extracted page by page

2. **Text Processing**
   - Extracted text is cleaned and split into semantic chunks using NLTK
   - Each chunk is tagged with its page number

3. **Embedding Generation**
   - Sentence-transformer model converts chunks into dense vector embeddings

4. **Vector Storage**
   - FAISS index stores embeddings for fast similarity search

5. **Query Handling**
   - User question is embedded
   - FAISS retrieves the most relevant document chunks

6. **Answer Generation**
   - Retrieved context is passed to the language model
   - Model generates a grounded answer using only retrieved content
   - Page citations are appended to the response

7. **Chat Memory**
   - Previous questions and answers are preserved to maintain context

---

## Technologies Used
- Python
- Gradio (UI)
- PyPDF
- NLTK
- Sentence-Transformers
- FAISS
- Hugging Face Transformers
- Hugging Face Spaces (Deployment)

---

