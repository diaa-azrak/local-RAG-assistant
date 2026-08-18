# 🤖 Offline Local RAG Assistant

A lightweight, on-device Retrieval-Augmented Generation (RAG) assistant built with Microsoft Foundry Local, Python, SQLite, Qwen models, and Streamlit.

The project demonstrates how a local document collection can be converted into embeddings, searched using semantic similarity, and provided as context to a locally running Large Language Model (LLM).

---

## 📌 Project Overview

The Offline Local RAG Assistant is designed to answer questions using information retrieved from a local knowledge base.

The system follows the RAG pipeline:

```text
Local Documents
      ↓
Paragraph Chunking
      ↓
qwen3-embedding-0.6b
      ↓
1024-Dimensional Embeddings
      ↓
SQLite
      ↓
User Question
      ↓
Query Embedding
      ↓
Cosine Similarity Search
      ↓
Relevant Context
      ↓
Prompt + Context
      ↓
qwen2.5-0.5b
      ↓
Generated Answer
