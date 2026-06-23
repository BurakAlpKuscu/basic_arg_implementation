# Basic RAG Implementation & Chunking Experiments

## 📌 Project Overview

This project is a **hands-on exploration of Retrieval-Augmented Generation (RAG)** systems.
It focuses on understanding and comparing different **text chunking strategies**, building a **vector search pipeline**, and enhancing retrieval quality using a **reranker model** before passing context to an LLM.

The final system allows querying a local document and getting context-aware answers using a local LLM (Ollama + Qwen).

---

## ⚙️ What This Project Does

* Loads a local text dataset
* Splits text using multiple chunking strategies
* Converts chunks into embeddings using Sentence Transformers
* Stores embeddings in FAISS (HNSW index)
* Retrieves relevant chunks based on user query
* Reranks retrieved chunks using a Cross-Encoder model
* Sends final context to a local LLM (Qwen3 via Ollama)

---

## 🧪 Chunking Experiments

Before building the final RAG pipeline, multiple chunking strategies were tested:

### 1. Size-Based Chunking

* Splits text into fixed-size character windows
* Simple but may break semantic structure

### 2. Sentence + Overlap Chunking

* Splits by sentences
* Adds overlap between chunks for context continuity

### 3. Semantic Chunking

* Uses sentence embeddings to group semantically similar sentences
* Dynamic threshold based on similarity distribution

### 4. RecursiveCharacterTextSplitter (LangChain)

* Final chosen method
* Splits text using hierarchical separators:

  * Paragraph → sentence → word → character
* Best balance between structure and consistency

---

## 🧠 Final RAG Pipeline

The final architecture used in this project:

```
Input Query
    ↓
Embedding (BAAI/bge-base-en-v1.5)
    ↓
FAISS HNSW Retrieval (Top-K=10)
    ↓
Cross-Encoder Reranker (BAAI/bge-reranker-base)
    ↓
Top-3 Relevant Chunks
    ↓
Context Injection
    ↓
Qwen3:4B (Ollama)
    ↓
Answer Generation
```

---

## 🧩 Key Components

### 🔹 Embedding Model

* `BAAI/bge-base-en-v1.5`
* Produces dense vector representations of text chunks

### 🔹 Vector Database

* FAISS (HNSW Index)
* Fast approximate nearest neighbor search

### 🔹 Reranker

* `BAAI/bge-reranker-base`
* Improves retrieval precision by scoring query–chunk pairs

### 🔹 LLM

* Qwen3:4B (via Ollama)
* Generates final answers using retrieved context

---

## 📊 Key Insights

During experimentation, the following observations were made:

* Recursive chunking outperformed semantic chunking for structured text
* Semantic chunking produced inconsistent chunk sizes
* Reranking significantly improved retrieval quality
* FAISS alone is not sufficient for high-precision QA systems
* Hybrid approach (retrieval + rerank) provides best results

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama

```bash
ollama serve
ollama pull qwen3:4b
```

### 3. Run pipeline

```bash
python rag_pipeline.py
```

## 🧠 Summary

This project demonstrates a full RAG pipeline from scratch, including:

* Multiple chunking strategy experiments
* Dense retrieval with FAISS
* Reranking for improved precision
* Local LLM inference with Ollama

The goal was not just to build a working system, but to understand **why each component matters in a retrieval-augmented architecture**.
