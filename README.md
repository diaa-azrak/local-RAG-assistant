# 🤖 Offline Local RAG Assistant

> **A 100% Private, On-Device Retrieval-Augmented Generation (RAG) System built with Microsoft Foundry Local, SQLite, and Streamlit.**

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Runtime](https://img.shields.io/badge/Runtime-Microsoft%20Foundry%20Local-purple.svg)
![Database](https://img.shields.io/badge/Database-SQLite%20Vector%20Store-green.svg)
![UI](https://img.shields.io/badge/UI-Streamlit-red.svg)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline-brightgreen.svg)

---

## 📌 Executive Summary

The **Offline Local RAG Assistant** is an enterprise-grade, privacy-first question-answering system that operates completely offline with **zero cloud dependencies, external API calls, or data leakage risks**. 

By combining **Microsoft Foundry Local** for hardware-accelerated local inference, a lightweight **SQLite vector engine** for persistence, and **strict prompt guardrails**, this assistant accurately grounds Large Language Model (LLM) responses in user-provided documents while eliminating hallucinations.

---

## ✨ Key Features

* 🔒 **100% Offline & Private:** Operates entirely on-device (CPU/GPU/NPU). No API keys or active internet connection required after initial setup.
* ⚡ **Lightweight Vector Storage:** Custom SQLite vector engine storing raw passage chunks alongside 1024-dimensional embeddings.
* 🎯 **Strict Hallucination Prevention:** Hardened system prompt guardrails enforce strict context adherence, falling back safely when information is missing.
* 📚 **Transparent Source Citation:** Displays real-time cosine similarity scores and exact reference passages for every generated answer.
* 🎨 **Dual Mode Access:** Includes both an interactive Command Line Interface (CLI) and a sleek Streamlit Web Interface.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data Ingestion Pipeline
        A[Local Documents .txt] -->|Paragraph Chunking| B[Text Chunks]
        B -->|qwen3-embedding-0.6b| C[1024-Dim Vector Embeddings]
        C -->|Persist Data| D[(SQLite Database rag_knowledge_base.db)]
    end

    subgraph Query & Retrieval Pipeline
        E[User Question] -->|Generate Query Vector| F[Query Embedding]
        F -->|Cosine Similarity Search| D
        D -->|Top K Scored Matches| G[Retrieved Context Passages]
    end

    subgraph Augmented Generation Pipeline
        G -->|Inject Context & Rules| H[Hardened System Prompt]
        E --> H
        H -->|qwen2.5-0.5b Chat LLM| I[Local On-Device Inference]
        I --> J[Grounded Answer + Source Citations]
    end