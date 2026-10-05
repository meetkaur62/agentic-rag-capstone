"""Knowledge store: embeddings + ChromaDB vector search + BM25 keyword search + re-ranking.

Retrieval is *hybrid*:
  1. Dense search (BGE embeddings in ChromaDB) finds passages with similar meaning.
  2. Sparse search (BM25) finds passages with the exact words (IDs, names, numbers).
  3. A cross-encoder re-ranker (BGE reranker) scores every candidate against the
     question and keeps only the most relevant ones above a relevance threshold.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .chunking import Chunk
from .config import settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    score: float  # final relevance score (re-ranker probability when available)

    def citation(self) -> str:
        m = self.metadata
        loc = ""
        if m.get("page"):
            loc = f", page {m['page']}"
        elif m.get("rows"):
            sheet = f", sheet '{m['sheet']}'" if m.get("sheet") else ""
            loc = f"{sheet}, rows {m['rows']}"
        return f"{m.get('source', 'unknown')}{loc}"


# --------------------------------------------------------------------------- #
# Models (lazy-loaded once, shared)
# --------------------------------------------------------------------------- #
class Embedder:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name or settings.embedding_model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 64).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(settings.embedding_query_prefix + text,
                                 normalize_embeddings=True).tolist()


class Reranker:
    """Cross-encoder re-ranker with a start-up self-test.

    A re-ranker that silently mis-scores (wrong library version, bad weights) is worse than
    none, so each candidate model must rank an obviously relevant passage above an
    irrelevant one before it is used.
    """

    SELF_TEST = [
        ("How many subsidiaries does the company have?",
         "As on March 31, 2026, we have 30 direct subsidiaries and 84 step-down subsidiaries.",
         "Segment footnote: Hi-Tech includes enterprises in Hi-Tech; LS includes Life Sciences and Healthcare."),
        ("How many days of paternity leave do employees get?",
         "Paternity leave is 10 working days, to be taken within 3 months of the child's birth.",
         "Delivery is free on orders above Rs. 499."),
    ]

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or settings.reranker_model
        self.model = CrossEncoder(self.model_name, max_length=512)

    def _raw(self, pairs):
        import numpy as np

        return np.asarray(self.model.predict(pairs, show_progress_bar=False), dtype=float).reshape(-1)

    def score(self, query: str, texts: list[str]) -> list[float]:
        import numpy as np

        raw = self._raw([(query, t) for t in texts])
        if raw.min() < 0 or raw.max() > 1:  # logits -> probabilities
            raw = 1 / (1 + np.exp(-raw))
        return raw.tolist()

    def self_test(self) -> bool:
        margins = []
        for q, good, bad in self.SELF_TEST:
            g, b = self.score(q, [good, bad])
            margins.append(g - b)
        ok = all(m > 0.2 for m in margins)
        logger.info("Re-ranker %s self-test margins %s -> %s", self.model_name,
                    [round(m, 3) for m in margins], "OK" if ok else "FAILED")
        return ok


def load_reranker() -> Reranker | None:
    """Return the first re-ranker that passes its self-test, else None (fused ranking is used)."""
    for name in dict.fromkeys([settings.reranker_model] + settings.reranker_fallbacks):
        try:
            rr = Reranker(name)
            if rr.self_test():
                return rr
            logger.warning("Re-ranker %s failed its self-test; trying the next one.", name)
        except Exception as e:
            logger.warning("Re-ranker %s could not be loaded (%s).", name, e)
    logger.warning("No working re-ranker; using hybrid fused ranking only.")
    return None


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class KnowledgeStore:
    def __init__(self, embedder=None, reranker=None, persist_dir: Path | None = None):
        import chromadb

        self.persist_dir = Path(persist_dir or settings.chroma_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir = self.persist_dir.parent / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        from chromadb.config import Settings as ChromaSettings

        self.client = chromadb.PersistentClient(path=str(self.persist_dir),
                                                settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            settings.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = embedder or Embedder()
        self.reranker = reranker
        if self.reranker is None:
            self.reranker = load_reranker()

        self._lock = threading.Lock()
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._bm25_docs: dict[str, tuple[str, dict]] = {}
        self.tables: dict[str, pd.DataFrame] = {}
        self._load_tables()
        self._rebuild_bm25()

    # ---------------- persistence helpers ----------------
    def _table_path(self, table_name: str) -> Path:
        safe = re.sub(r"[^\w.-]+", "_", table_name)
        return self.tables_dir / f"{safe}.pkl"

    def _load_tables(self) -> None:
        index = self.tables_dir / "index.json"
        if not index.exists():
            return
        for name, fname in json.loads(index.read_text()).items():
            p = self.tables_dir / fname
            if p.exists():
                self.tables[name] = pd.read_pickle(p)

    def _save_table_index(self) -> None:
        index = {name: self._table_path(name).name for name in self.tables}
        (self.tables_dir / "index.json").write_text(json.dumps(index, indent=2))

    def _rebuild_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        data = self.collection.get(include=["documents", "metadatas"])
        self._bm25_ids = data["ids"]
        self._bm25_docs = {i: (d, m) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])}
        corpus = [tokenize(d) for d in data["documents"]]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    # ---------------- document management ----------------
    def has_document(self, file_hash: str) -> bool:
        return bool(self.collection.get(where={"file_hash": file_hash}, limit=1)["ids"])

    def add_chunks(self, chunks: list[Chunk], tables: dict[str, pd.DataFrame] | None = None) -> int:
        if not chunks:
            return 0
        with self._lock:
            batch = 256
            for i in range(0, len(chunks), batch):
                part = chunks[i:i + batch]
                texts = [c.text for c in part]
                self.collection.upsert(
                    ids=[c.id for c in part],
                    documents=texts,
                    embeddings=self.embedder.embed_documents(texts),
                    metadatas=[{k: v for k, v in c.metadata.items() if v is not None} for c in part],
                )
            for name, df in (tables or {}).items():
                self.tables[name] = df
                df.to_pickle(self._table_path(name))
            self._save_table_index()
            self._rebuild_bm25()
        return len(chunks)

    def list_documents(self) -> list[dict]:
        data = self.collection.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for m in data["metadatas"]:
            d = docs.setdefault(m["source"], {"source": m["source"], "type": m.get("doc_type", ""), "chunks": 0})
            d["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["source"])

    def delete_document(self, source: str) -> None:
        with self._lock:
            self.collection.delete(where={"source": source})
            for name in [t for t in self.tables if t == source or t.startswith(source + "::")]:
                self._table_path(name).unlink(missing_ok=True)
                del self.tables[name]
            self._save_table_index()
            self._rebuild_bm25()

    def reset(self) -> None:
        with self._lock:
            self.client.delete_collection(settings.collection_name)
            self.collection = self.client.get_or_create_collection(
                settings.collection_name, metadata={"hnsw:space": "cosine"}
            )
            for p in self.tables_dir.glob("*"):
                p.unlink()
            self.tables.clear()
            self._rebuild_bm25()

    def count(self) -> int:
        return self.collection.count()

    # ---------------- retrieval ----------------
    def _dense_search(self, query: str, k: int) -> list[str]:
        n = self.collection.count()
        if n == 0:
            return []
        res = self.collection.query(query_embeddings=[self.embedder.embed_query(query)], n_results=min(k, n))
        return res["ids"][0]

    def _sparse_search(self, query: str, k: int) -> list[str]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._bm25_ids[i] for i in ranked[:k] if scores[i] > 0]

    def search(self, queries: list[str], rerank_query: str, top_k: int | None = None,
               min_score: float | None = None) -> list[RetrievedChunk]:
        """Hybrid search for one or more query variants, re-ranked against `rerank_query`."""
        top_k = top_k or settings.final_top_k
        min_score = settings.min_relevance if min_score is None else min_score

        # Reciprocal Rank Fusion over every (query, retriever) result list
        fused: dict[str, float] = {}
        for q in queries:
            for ranked in (self._dense_search(q, settings.dense_top_k),
                           self._sparse_search(q, settings.sparse_top_k)):
                for rank, cid in enumerate(ranked):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
        if not fused:
            return []

        n_cand = 30 if _has_gpu() else 15  # cross-encoders are slow on CPU-only hosts
        candidates = sorted(fused, key=fused.get, reverse=True)[:n_cand]
        texts = [self._bm25_docs[c][0] for c in candidates]

        if self.reranker is not None:
            scores = self.reranker.score(rerank_query, texts)
            best = max(fused.values())
            ranked = []
            for rank, (c, rr) in enumerate(zip(candidates, scores)):
                # Safety net: a passage that BOTH vector and keyword search rank at the very top
                # is kept even if the re-ranker under-scores it (at a much lower bar).
                consensus = rank < 3 and rr >= min_score / 10
                if rr >= min_score or consensus:
                    ranked.append((c, 0.8 * rr + 0.2 * fused[c] / best))
            ranked.sort(key=lambda x: x[1], reverse=True)
        else:  # no re-ranker: normalise fused scores to 0-1 and keep the top ones
            best = max(fused.values())
            ranked = [(c, fused[c] / best) for c in candidates]

        return [RetrievedChunk(c, self._bm25_docs[c][0], self._bm25_docs[c][1], round(float(s), 4))
                for c, s in ranked[:top_k]]
