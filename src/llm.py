"""LLM client with automatic failover: Groq (primary) -> Gemini (backup).

Both providers expose OpenAI-compatible endpoints, so one SDK (`openai`) talks to
both. Each agent role has its own Groq model, which spreads requests across the
per-model free-tier rate limits.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from .config import settings

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """No provider could answer (missing keys, all rate-limited, or network down)."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


class LLMClient:
    ROLE_MODELS = {
        "planner": "planner_model",
        "reasoner": "reasoner_model",
        "validator": "validator_model",
    }
    ROLE_EFFORT = {"planner": "low", "reasoner": "medium", "validator": "low"}

    def __init__(self):
        from openai import OpenAI

        self._groq = self._gemini = None
        if settings.groq_api_key:
            self._groq = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url,
                                timeout=settings.llm_timeout_s, max_retries=0)
        if settings.gemini_api_key:
            self._gemini = OpenAI(api_key=settings.gemini_api_key, base_url=settings.gemini_base_url,
                                  timeout=settings.llm_timeout_s, max_retries=0)

    # ------------------------------------------------------------------ #
    @property
    def available_providers(self) -> list[str]:
        return [n for n, c in (("Groq", self._groq), ("Gemini", self._gemini)) if c]

    def _attempts(self, role: str):
        if self._groq:
            yield "Groq", self._groq, getattr(settings, self.ROLE_MODELS[role])
        if self._gemini:
            for m in settings.gemini_models:
                yield "Gemini", self._gemini, m

    @staticmethod
    def _call(client, model: str, messages: list[dict], max_tokens: int, effort: str | None) -> str:
        kwargs = dict(model=model, messages=messages, temperature=0, max_tokens=max_tokens)
        if effort:
            kwargs["reasoning_effort"] = effort
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:  # some models reject reasoning_effort -> retry without it
            if effort and "reasoning" in str(e).lower():
                kwargs.pop("reasoning_effort")
                resp = client.chat.completions.create(**kwargs)
            else:
                raise
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("empty response")
        return text

    def chat(self, role: str, messages: list[dict], max_tokens: int = 1500) -> LLMResponse:
        if not self.available_providers:
            raise LLMUnavailableError(
                "No API key found. Add GROQ_API_KEY (and optionally GEMINI_API_KEY) to Colab Secrets "
                "or your environment."
            )
        errors = []
        for provider, client, model in self._attempts(role):
            effort = self.ROLE_EFFORT.get(role)
            for attempt in range(settings.llm_max_retries):
                try:
                    text = self._call(client, model, messages, max_tokens, effort)
                    return LLMResponse(text, provider, model)
                except Exception as e:
                    msg = str(e)
                    errors.append(f"{provider}/{model}: {msg[:160]}")
                    logger.warning("LLM call failed (%s/%s, attempt %d): %s", provider, model, attempt + 1, msg[:200])
                    status = getattr(e, "status_code", None)
                    if status in (400, 401, 403, 404):  # bad request / key / model: don't retry this model
                        break
                    if status == 429 and attempt == 0:   # rate-limited: brief back-off, then one retry
                        time.sleep(3)
                    elif attempt == 0:
                        time.sleep(1)
        raise LLMUnavailableError("All LLM providers failed: " + " | ".join(errors[-4:]))

    def chat_json(self, role: str, messages: list[dict], default: dict, max_tokens: int = 1000) -> tuple[dict, LLMResponse | None]:
        """Ask for a JSON object and parse it robustly; return `default` if parsing fails."""
        resp = self.chat(role, messages, max_tokens)
        parsed = parse_json(resp.text)
        if parsed is None:
            logger.warning("Could not parse JSON from %s: %s", role, resp.text[:200])
            return dict(default), resp
        return parsed, resp

    # ------------------------------------------------------------------ #
    def prompt_injection_score(self, text: str) -> float | None:
        """Probability (0-1) that `text` is a jailbreak / prompt-injection attempt.

        Uses Meta's Llama Prompt Guard 2 on Groq. Returns None if unavailable,
        in which case only the local rule-based checks apply.
        """
        if not self._groq or getattr(self, "_guard_disabled", False):
            return None
        if not getattr(self, "_guard_checked", False):
            # Self-check once: a clearly benign text must score low. If it doesn't
            # (unexpected output format), disable the model check rather than block everyone.
            self._guard_checked = True
            benign = self._raw_guard_score("What is the annual leave policy?")
            if benign is None or benign >= 0.5:
                logger.warning("Prompt Guard self-check failed (benign score=%s); using rule-based checks only.", benign)
                self._guard_disabled = True
                return None
        return self._raw_guard_score(text)

    def _raw_guard_score(self, text: str) -> float | None:
        try:
            resp = self._groq.chat.completions.create(
                model=settings.prompt_guard_model,
                messages=[{"role": "user", "content": text[:2000]}],
                temperature=0,
            )
            out = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\d*\.?\d+(?:[eE]-?\d+)?", out)
            if m:
                return max(0.0, min(1.0, float(m.group())))
            low = out.lower()
            if any(w in low for w in ("malicious", "jailbreak", "injection", "unsafe")):
                return 1.0
            if any(w in low for w in ("benign", "safe")):
                return 0.0
        except Exception as e:
            logger.warning("Prompt guard unavailable: %s", str(e)[:200])
        return None


def parse_json(text: str) -> dict | None:
    """Extract the first JSON object from an LLM reply (handles ```json fences and extra text)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
