## How it works

1. **Upload**: files are validated (type, size, encryption, empty content) and parsed. PDFs are read in the right order even with two columns, their tables are kept as tables, and scanned pages are OCR'd. Then everything is split into overlapping chunks.
2. **Index**: each chunk is embedded with `BAAI/bge-base-en-v1.5` and stored in ChromaDB. A BM25 keyword index is built alongside it. CSV and Excel tables are also kept as DataFrames for exact calculations.
3. **Ask**: your question passes through five steps:

| Step | What it does |
|---|---|
| 🛡️ **Guard** | Blocks empty, oversized, prompt-injection (rules + Llama Prompt Guard 2) and harmful requests |
| 🧭 **Planner** | Understands intent, rewrites follow-ups, writes search queries, picks tools |
| 🔎 **Retriever** | Hybrid vector + keyword search, cross-encoder re-ranking, relevance threshold, table tool |
| ✍️ **Reasoner** | Answers only from the numbered sources and cites every statement |
| ✅ **Validator** | Fact-checks the answer; unsupported claims trigger one rewrite, then a low-confidence warning |

If nothing relevant is found, the system says so instead of guessing.

**Models (all free):** Groq `openai/gpt-oss-120b` and `openai/gpt-oss-20b`, with Gemini Flash as backup · BGE embeddings and BGE v2-m3 re-ranker from Hugging Face.
