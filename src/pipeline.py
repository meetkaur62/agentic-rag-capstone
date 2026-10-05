"""High-level facade used by the UI, the notebook and the tests."""
from __future__ import annotations

import logging
from pathlib import Path

from .agents import AgentWorkflow
from .chunking import chunk_document
from .ingestion import IngestionError, load_document
from .llm import LLMClient, LLMUnavailableError
from .vector_store import KnowledgeStore

logger = logging.getLogger(__name__)


class KnowledgeAssistant:
    def __init__(self, store: KnowledgeStore | None = None, llm: LLMClient | None = None):
        self.store = store or KnowledgeStore()
        self.llm = llm or LLMClient()
        self.workflow = AgentWorkflow(self.store, self.llm)

    # ---------------- ingestion ----------------
    def ingest(self, paths: list[str | Path], display_names: list[str] | None = None) -> list[dict]:
        """Ingest files; never raises. Returns one status row per file."""
        report = []
        for i, path in enumerate(paths):
            path = Path(path)
            name = display_names[i] if display_names else path.name
            try:
                doc = load_document(path, display_name=name)
                if self.store.has_document(doc.file_hash):
                    report.append({"file": name, "status": "skipped", "detail": "Already indexed (identical file)."})
                    continue
                if any(d["source"] == name for d in self.store.list_documents()):
                    self.store.delete_document(name)  # a new version replaces the old one
                chunks = chunk_document(doc)
                n = self.store.add_chunks(chunks, doc.tables)
                detail = f"{len(doc.sections)} sections -> {n} chunks"
                if doc.tables:
                    detail += f"; {len(doc.tables)} table(s) available for exact calculations"
                if doc.warnings:
                    detail += ". Note: " + " ".join(doc.warnings)
                report.append({"file": name, "status": "indexed", "detail": detail})
            except IngestionError as e:
                report.append({"file": name, "status": "rejected", "detail": str(e)})
            except Exception as e:  # unexpected: log full trace, show short message
                logger.exception("Failed to ingest %s", name)
                report.append({"file": name, "status": "error", "detail": f"Unexpected error: {e}"})
        return report

    # ---------------- Q&A ----------------
    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> dict:
        try:
            return self.workflow.run(question, history)
        except LLMUnavailableError as e:
            return {"answer": f"The language model is unavailable right now. {e}", "status": "error",
                    "sources": [], "trace": [], "models_used": [], "latency_s": 0}
        except Exception as e:
            logger.exception("Question failed")
            return {"answer": f"Something went wrong while answering: {e}", "status": "error",
                    "sources": [], "trace": [], "models_used": [], "latency_s": 0}

    # ---------------- management ----------------
    def documents(self) -> list[dict]:
        return self.store.list_documents()

    def delete(self, source: str) -> None:
        self.store.delete_document(source)

    def reset(self) -> None:
        self.store.reset()
