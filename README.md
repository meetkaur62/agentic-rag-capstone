---
title: Agentic RAG Document Assistant
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
---

# 🧠 Agentic RAG – Enterprise Document Assistant

A Generative AI application for asking questions about enterprise documents. Upload **PDF, TXT, CSV or Excel**
files, ask questions in plain English, and four cooperating AI agents return answers **grounded in your documents, with citations**.

Built entirely on **free** resources: open-source models from Hugging Face and the free tiers of Groq and Google Gemini.

---

## ✨ Features

| Capability | How |
|---|---|
| Multi-format ingestion | PDF (layout-aware with PyMuPDF4LLM: multi-column reading order, tables kept as Markdown tables, OCR for scanned pages via Tesseract), TXT (encoding detection), CSV (delimiter sniffing), Excel (every sheet) |
| Semantic chunking | Recursive splitting on paragraph → sentence → word boundaries, with overlap. Tables are split only between rows, and each piece repeats the header row and section title |
| Vector knowledge store | `BAAI/bge-base-en-v1.5` embeddings in **ChromaDB** (persistent, cosine similarity) |
| Hybrid retrieval | Dense vector search **+ BM25 keyword search**, merged with Reciprocal Rank Fusion |
| Re-ranking | `BAAI/bge-reranker-v2-m3` cross-encoder (self-tested at start-up, with automatic fallback to smaller re-rankers), plus a relevance threshold |
| Agentic reasoning | **LangGraph** workflow: Guard → Planner → Retriever → Reasoner → Validator (with self-correction loop) |
| Exact numbers | A safe **table-analysis tool**: the LLM fills a JSON spec and pandas computes the answer, so no code is executed |
| Anti-hallucination | Answers only from sources, mandatory citations, validator fact-check, "not found" instead of guessing |
| Safety | Input validation, rule-based and **Llama Prompt Guard 2** injection detection, harmful-request filter, secret redaction |
| Reliability | Automatic failover from Groq to Gemini, retries with back-off, graceful errors, duplicate-upload detection |
| Transparency | The UI shows the sources used and a step-by-step agent trace for every answer |

## 🏗️ Architecture

```
            ┌──────────────── Ingestion ────────────────┐
 Upload ──► │ validate → parse (PDF/TXT/CSV/XLSX) → chunk │──► embeddings (BGE) ──► ChromaDB
            │                     └─► DataFrames ─────────┼──► Table store (pandas)
            └─────────────────────────────────────────────┘          + BM25 index

 Question ─► [Guard] ─► [Planner] ─► [Retriever] ─► [Reasoner] ─► [Validator] ─► Answer + citations
               │            │            │  hybrid search          │   ▲             │
               │ blocked    │ chit-chat  │  + re-rank              │   └─ rejected ──┘ (retry once
               ▼            ▼            │  + table tool           │      with feedback)
            refusal     direct reply     └─ nothing relevant ─► "Not found in documents"
```

### Agent roles

| Agent | Model | Responsibility |
|---|---|---|
| **Guard** | rules + `llama-prompt-guard-2-86m` | Validates input; blocks prompt-injection and harmful requests |
| **Planner** | `openai/gpt-oss-20b` | Classifies intent, rewrites follow-up questions, writes search queries, chooses tools (document search and/or table analysis) |
| **Retriever** | BGE + BM25 + BGE re-ranker (v2-m3) | Hybrid search, re-ranking, relevance filtering; runs the table tool |
| **Reasoner** | `openai/gpt-oss-120b` | Writes a concise answer from the numbered sources only, citing every claim |
| **Validator** | `openai/gpt-oss-20b` | Fact-checks each claim against the sources; rejects and requests a rewrite if unsupported |

All LLM calls fall back to **Gemini** (`gemini-3.5-flash`, then `gemini-3.5-flash-lite`) if Groq fails or is rate-limited.
Model names live in `src/config.py` and can be overridden with environment variables.

> Note: Groq retired Llama 3.3 70B on 16 Aug 2026. `openai/gpt-oss-120b` is Groq's recommended replacement.

## 📁 Project structure

```
agentic-rag-capstone/
├── app.py                  # Gradio UI (also the Hugging Face Spaces entry point)
├── requirements.txt
├── packages.txt            # system package for Hugging Face Spaces (Tesseract OCR)
├── .env.example            # template for API keys (never commit real keys)
├── src/
│   ├── config.py           # all settings: models, chunk sizes, thresholds, limits
│   ├── ingestion.py        # file validation + PDF/TXT/CSV/Excel loaders
│   ├── chunking.py         # recursive chunking with overlap
│   ├── vector_store.py     # embeddings, ChromaDB, BM25, re-ranker, hybrid search
│   ├── llm.py              # Groq → Gemini client with retries/failover, Prompt Guard
│   ├── tools.py            # safe table-analysis tool (pandas)
│   ├── guardrails.py       # input/output safety checks
│   ├── agents.py           # LangGraph multi-agent workflow + prompts
│   └── pipeline.py         # high-level KnowledgeAssistant facade
├── notebooks/Capstone_Colab.ipynb   # run everything in Google Colab
├── scripts/
│   ├── make_sample_docs.py # creates the demo documents
│   └── evaluate.py         # end-to-end evaluation with real models
├── sample_docs/            # demo PDF, TXT, CSV, XLSX (fictional company)
└── tests/test_pipeline.py  # 27 offline tests (no API keys needed)
```

## 🚀 Setup

### 1. Get free API keys
* **Groq** (required): <https://console.groq.com> → API Keys → Create
* **Gemini** (optional backup): <https://aistudio.google.com> → Get API key

### 2a. Google Colab (easiest)
1. Open `notebooks/Capstone_Colab.ipynb` in Colab. Runtime → Change runtime type → **T4 GPU** (optional; makes it faster).
2. Click the 🔑 **Secrets** icon and add `GROQ_API_KEY` (and `GEMINI_API_KEY`). Turn on **Notebook access**.
3. Run the cells in order. The last cell prints a public `https://….gradio.live` link to the app.

### 2b. Local machine
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then paste your keys into .env
python app.py               # open http://127.0.0.1:7860
```

### 3. Tests and evaluation
```bash
python -m pytest tests -q          # offline tests, no keys needed
python scripts/evaluate.py         # real end-to-end evaluation on sample_docs (needs GROQ_API_KEY)
```

## ☁️ Deployment (Hugging Face Spaces, free)
1. Create a new Space at <https://huggingface.co/new-space>, choose **Gradio** SDK and the free **CPU basic** hardware.
2. Upload every file in this folder (the YAML block at the top of this README configures the Space).
3. Space → **Settings → Variables and secrets** → add `GROQ_API_KEY` and `GEMINI_API_KEY` as **secrets**.
4. The Space builds automatically, and the app is live at `https://huggingface.co/spaces/<user>/<space>`.

## ⚠️ Limitations
* **Scanned PDFs:** OCR (Tesseract) works for clean scans but is slower, and handwriting or poor-quality scans may be misread. Standalone image files (JPG/PNG) are not supported.
* **Complex PDF tables:** merged or multi-level headers can come out slightly garbled, so check numbers against the cited page.
* **Free-tier limits:** Groq's free plan allows about 8K tokens per minute per model, so heavy use can trigger brief rate limits. The Gemini fallback covers most of these cases.
* **Hallucination is reduced, not eliminated:** the validator is itself an LLM. Always check the cited sources for critical decisions.
* **Table tool scope:** single-table operations only (filter, aggregate, group-by, top-N). No joins across tables.
* **Storage on free Spaces is temporary:** uploaded documents are lost when the Space restarts.
* **English-focused:** the BGE models are English models. Other languages work less well.
* **No user accounts:** every user of one deployment shares the same document store.
