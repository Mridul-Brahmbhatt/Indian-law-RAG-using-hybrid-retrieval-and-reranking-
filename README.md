# Indian Constitution Legal Case Evaluation System

An AI-powered legal analysis system built using **LangChain**, **Ollama**, **Hybrid Retrieval**, and **LLM-based Reranking** to evaluate legal cases based on the **Indian Constitution** and provide judgment-style responses.

This project combines **semantic search + keyword search + contextual reranking** to improve retrieval quality and generate more legally relevant answers.

---

## Features

-  AI-powered legal case evaluation
-  Uses the **Indian Constitution PDF** as legal knowledge base
-  **Hybrid Retrieval**
  - Semantic vector search using embeddings
  - Keyword-based BM25 retrieval
-  **LLM-based Reranking / Context Compression**
  - Filters and reranks retrieved legal context
  - Improves relevance before sending context to the LLM
-  Generates:
  - Guilty judgment analysis
  - Innocent judgment analysis
  - Probability/likelihood assessment
  - Lawyer-style advice
-  Persistent vector database using ChromaDB
-  Built with LangChain modular architecture

---

# Why Hybrid Retrieval?

Traditional retrieval methods often fail in legal systems because:

- Keyword search may miss semantic meaning
- Semantic search may ignore exact legal terminology

This project solves that by combining both approaches.

## Hybrid Retrieval = BM25 + Vector Search

### 1. BM25 Retrieval

Finds documents using:
- Exact legal terms
- Constitutional articles
- Important legal keywords

Example:
- “Article 21”
- “Right to Equality”
- “Murder”
- “Bail”

### 2. Semantic Vector Retrieval

Uses embeddings to understand:
- Context
- Meaning
- Intent of the case

This helps retrieve legally relevant sections even when wording differs.

---

# Why Reranking / Context Compression?

Even after retrieval, some chunks may still be:
- partially relevant
- noisy
- repetitive

The project uses an **LLM-based Contextual Compression Retriever** to:
- analyze retrieved chunks
- remove irrelevant information
- keep only the most useful legal context

## Benefits of Reranking

- Better legal reasoning
- More accurate judgment generation
- Reduced hallucinations
- Higher relevance of constitutional references
- Cleaner prompts for the LLM

---

# Tech Stack

- Python
- LangChain
- Ollama
- ChromaDB
- BM25 Retriever
- Ensemble Retriever
- Contextual Compression Retriever
- Llama 3.2
- Qwen 2.5
- MXBAI Embeddings

---

# Project Architecture

```text
User Case Input
       ↓
Hybrid Retrieval
(BM25 + Vector Search)
       ↓
LLM Reranking / Compression
       ↓
Filtered Legal Context
       ↓
LLM Legal Analysis
       ↓
Judgment-style Response
```

---

# Dataset / Legal Source

The project uses:

- `20240716890312078.pdf`

This PDF contains the legal and constitutional knowledge base used for retrieval and analysis.

---


# Project Structure

```text
project/
│
├── data/
│   └── 20240716890312078.pdf
│
├── chroma_langchain_db/
│
├── main.py
├── vector.py
└── README.md
```

---

# How It Works

## Step 1 — PDF Processing

The constitution PDF is:
- loaded
- split into chunks
- converted into embeddings

using:
- `PyPDFLoader`
- `RecursiveCharacterTextSplitter`
- `OllamaEmbeddings`

---

## Step 2 — Hybrid Retrieval

The system combines:
- BM25 keyword retrieval
- Chroma vector retrieval

using LangChain's `EnsembleRetriever`.

---

## Step 3 — LLM Reranking

Retrieved chunks are passed through:
- `LLMChainExtractor`
- `ContextualCompressionRetriever`

to keep only the most relevant legal context.

---

## Step 4 — Legal Judgment Generation

The final filtered legal context is sent to the LLM which:
- analyzes the case
- references relevant laws
- generates judgment-style outputs

---

# Example Usage

```bash
python main.py
```

Example input:

```text
A person was arrested without being informed of the reason for arrest and was denied legal representation.
```

Example output includes:
- Relevant constitutional principles
- Guilty analysis
- Innocent analysis
- Likelihood estimation
- Legal advice

---

# Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Hybrid Retrieval Systems
- Legal AI pipelines
- Vector Databases
- Embeddings
- LLM Reranking
- Prompt Engineering
- Context Compression
- AI-based Decision Support Systems

---
