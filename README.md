# 🧠 BrainGPT

A RAG-based Question Answering system for Brain Anatomy, built with **FastAPI**, **Streamlit**, **LangChain**, **FAISS**, and **Groq LLaMA3**.

---

## 🔍 Overview

BrainGPT allows users to ask natural language questions about brain anatomy.  
It retrieves relevant information from PDF documents using vector search (FAISS) and generates accurate answers using LLaMA-3.1 via Groq.

### Architecture

- **Backend**: FastAPI (handles document loading, embedding, retrieval & generation)
- **Frontend**: Streamlit (simple chat-like interface)
- **Embeddings**: Google Gemini Embeddings (`models/gemini-embedding-001`)
- **LLM**: Groq - `llama-3.1-8b-instant`
- **Vector Store**: FAISS

---

## 🚀 Features

- Load PDF documents and create a vector knowledge base
- Ask questions and get context-aware answers
- Clean separation of frontend and backend
- Persistent FAISS index (saved locally)
- Simple and fast Streamlit UI

---

## 📁 Project Structure

```bash
brain-gpt/
├── brain_docs/              # Put your PDF files here
│   └── brain_anatomy.pdf
├── main.py                  # FastAPI backend
├── frontend.py              # Streamlit frontend
├── pyproject.toml
├── requirements.txt
└── README.md