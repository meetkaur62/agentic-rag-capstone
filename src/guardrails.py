"""Reliability & safety controls.

Layers (cheapest first):
  1. Input validation      – empty / too long / control characters.
  2. Rule-based injection  – regex patterns for common jailbreak phrasing.
  3. Model-based injection – Llama Prompt Guard 2 (via Groq) probability score.
  4. Harmful-request check – refuse clearly dangerous requests.
  5. Output checks         – citations must point at real sources, secrets are redacted.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .config import settings

INJECTION_THRESHOLD = 0.8

_INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (all |any |the )?(previous|prior|above|system) (instructions|prompts|rules)",
    r"forget (all |your |the )?(previous |prior )?(instructions|rules)",
    r"(reveal|show|print|repeat|output) (me )?(your|the) (system prompt|hidden prompt|instructions|api key)",
    r"you are now (in )?(dan|developer mode|jailbroken|unrestricted)",
    r"\bjailbreak\b",
    r"act as (an? )?(unfiltered|unrestricted|uncensored)",
    r"pretend (that )?you (have no|don't have any) (rules|restrictions|guidelines)",
    r"bypass (your |the )?(safety|guardrails|filters|restrictions)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_HARMFUL_RE = re.compile(
    r"\b(how to|how do i|steps to|instructions (for|to))\b.{0,40}\b("
    r"make (a )?(bomb|explosive|nerve agent|bioweapon|meth)|build (a )?(bomb|weapon)|"
    r"kill (someone|a person|people)|hack into|steal (credit card|passwords?)|"
    r"write (a )?(malware|ransomware|keylogger))",
    re.IGNORECASE,
)

_SECRET_RE = re.compile(
    r"(gsk_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z\-_]{30,}|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})"
)


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""
    cleaned: str = ""


def sanitize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return re.sub(r"[ \t]+", " ", text).strip()


def check_question(question: str, llm=None) -> GuardResult:
    q = sanitize(question)
    if not q:
        return GuardResult(False, "Please type a question.")
    if len(q) < 2:
        return GuardResult(False, "The question is too short.")
    if len(q) > settings.max_question_chars:
        return GuardResult(False, f"Please keep questions under {settings.max_question_chars} characters.")
    if _INJECTION_RE.search(q):
        return GuardResult(False, "This looks like an attempt to override the assistant's instructions, so it was blocked.")
    if _HARMFUL_RE.search(q):
        return GuardResult(False, "I can't help with that request.")
    detail = "rule-based checks passed"
    if llm is not None:
        score = llm.prompt_injection_score(q)
        if score is not None and score >= INJECTION_THRESHOLD:
            return GuardResult(False, f"Blocked by the prompt-injection detector (risk score {score:.2f}).")
        detail += (f"; Prompt Guard risk score {score:.2f}" if score is not None
                   else "; Prompt Guard model unavailable")
    return GuardResult(True, reason=detail, cleaned=q)


def redact_secrets(text: str) -> str:
    return _SECRET_RE.sub("[REDACTED]", text)


_FANCY_CITE_RE = re.compile(r"【\s*(T?\d+)[^】]*】")          # gpt-oss style: 【1】 or 【1†L3-L5】
_GROUP_CITE_RE = re.compile(r"\[\s*((?:T?\d+\s*[,;]\s*)+T?\d+)\s*\]")  # [1, 2] -> [1][2]
_WORD_CITE_RE = re.compile(r"\[(?:source|src|doc)\s*(T?\d+)\]", re.IGNORECASE)  # [Source 1] -> [1]


def normalize_citations(answer: str) -> str:
    """Convert the different citation styles LLMs produce into the standard [n] form."""
    answer = _FANCY_CITE_RE.sub(r"[\1]", answer)
    answer = _WORD_CITE_RE.sub(r"[\1]", answer)
    answer = _GROUP_CITE_RE.sub(lambda m: "".join(f"[{x.strip()}]" for x in re.split(r"[,;]", m.group(1))), answer)
    return answer


def citation_ids(answer: str) -> set[str]:
    """All citation markers used in an answer, e.g. {'1', '3', 'T1'}."""
    return set(re.findall(r"\[(T?\d+)\]", answer))


def check_citations(answer: str, valid_ids: set[str]) -> tuple[bool, set[str]]:
    """Return (ok, invalid_ids). Invalid = cited a source number that doesn't exist."""
    used = citation_ids(answer)
    invalid = used - valid_ids
    return (bool(used) and not invalid), invalid


def strip_invalid_citations(answer: str, invalid: set[str]) -> str:
    for cid in invalid:
        answer = answer.replace(f"[{cid}]", "")
    return answer
