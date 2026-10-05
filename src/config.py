"""Central configuration.

Every model name, limit and path lives here so the system can be re-tuned
without touching the rest of the code. Values can be overridden with
environment variables (or Colab / Hugging Face secrets for API keys).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def get_secret(name: str) -> str | None:
    """Read a secret from env vars first, then from Google Colab Secrets."""
    value = os.getenv(name)
    if value:
        return value.strip()
    try:  # Only works inside Google Colab
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
        return value.strip() if value else None
    except Exception:
        return None


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else default


BASE_DIR = Path(__file__).resolve().parent.parent

try:  # local runs: read keys from a .env file if present
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass


@dataclass
class Settings:
    # ---------- LLM providers (both free tiers, OpenAI-compatible APIs) ----------
    groq_base_url: str = "https://api.groq.com/openai/v1"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Primary models on Groq. Llama 3.3 70B was retired by Groq on 16 Aug 2026;
    # Groq's recommended replacement is openai/gpt-oss-120b.
    reasoner_model: str = os.getenv("REASONER_MODEL", "openai/gpt-oss-120b")
    planner_model: str = os.getenv("PLANNER_MODEL", "openai/gpt-oss-20b")
    validator_model: str = os.getenv("VALIDATOR_MODEL", "openai/gpt-oss-20b")
    prompt_guard_model: str = os.getenv("PROMPT_GUARD_MODEL", "meta-llama/llama-prompt-guard-2-86m")

    # Fallback models on Gemini (tried in order if Groq fails / is rate-limited)
    gemini_models: list[str] = field(
        default_factory=lambda: _env_list("GEMINI_MODELS", ["gemini-3.5-flash", "gemini-3.5-flash-lite"])
    )

    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # ---------- Embeddings & re-ranking (free, open-source, Hugging Face) ----------
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    # tried in order if the main re-ranker fails to load or fails its self-test
    reranker_fallbacks: list[str] = field(default_factory=lambda: _env_list(
        "RERANKER_FALLBACKS", ["BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L-12-v2"]))

    # ---------- Chunking ----------
    chunk_size: int = 1000          # characters
    chunk_overlap: int = 150        # characters
    table_rows_per_chunk: int = 15  # rows of a CSV/Excel sheet per chunk

    # ---------- Retrieval ----------
    dense_top_k: int = 20           # candidates from vector search
    sparse_top_k: int = 20          # candidates from BM25 keyword search
    final_top_k: int = 6            # chunks passed to the LLM after re-ranking
    min_relevance: float = 0.05     # re-ranker score (0-1) below which a chunk is dropped

    # ---------- Agent loop ----------
    max_answer_attempts: int = 2    # reasoner retries when the validator rejects an answer

    # ---------- Input validation ----------
    allowed_extensions: tuple[str, ...] = (".pdf", ".txt", ".csv", ".xlsx", ".xls")
    max_file_mb: int = 25
    max_question_chars: int = 1000

    # ---------- Storage ----------
    data_dir: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    collection_name: str = "enterprise_docs"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def tables_dir(self) -> Path:
        return self.data_dir / "tables"

    @property
    def groq_api_key(self) -> str | None:
        return get_secret("GROQ_API_KEY")

    @property
    def gemini_api_key(self) -> str | None:
        return get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")


settings = Settings()
