"""Gradio user interface – Agentic RAG Enterprise Document Assistant.

Run locally:        python app.py
Run in Colab:       see notebooks/Capstone_Colab.ipynb (creates a public share link)
Hugging Face Space: this file is the entry point (README.md holds the Space config)
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import settings  # noqa: E402
from src.pipeline import KnowledgeAssistant  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "chromadb", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

_assistant: KnowledgeAssistant | None = None
_lock = threading.Lock()


def get_assistant() -> KnowledgeAssistant:
    global _assistant
    with _lock:
        if _assistant is None:
            _assistant = KnowledgeAssistant()
        return _assistant


STATUS_BADGE = {
    "ok": "✅ Verified against sources",
    "unverified": "⚠️ Not fully verified",
    "not_found": "🔍 Not found in documents",
    "blocked": "🛡️ Blocked by guardrails",
    "chitchat": "💬",
    "error": "❌ Error",
}


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def docs_table():
    return [[d["source"], d["type"], d["chunks"]] for d in get_assistant().documents()]


def doc_choices():
    return gr.update(choices=[d["source"] for d in get_assistant().documents()], value=None)


def upload(files):
    if not files:
        return [["-", "no files", "Please choose one or more files first."]], docs_table(), doc_choices()
    paths = [f if isinstance(f, str) else f.name for f in files]
    report = get_assistant().ingest(paths, [os.path.basename(p) for p in paths])
    rows = [[r["file"], r["status"], r["detail"]] for r in report]
    return rows, docs_table(), doc_choices()


def delete_doc(name):
    if name:
        get_assistant().delete(name)
    return docs_table(), doc_choices()


def reset_all():
    get_assistant().reset()
    return docs_table(), doc_choices(), [], []


def format_sources(sources):
    if not sources:
        return "_No sources for this answer._"
    out = []
    for s in sources:
        excerpt = s["text"].strip().replace("\n", " ")
        excerpt = excerpt[:350] + ("…" if len(excerpt) > 350 else "")
        out.append(f"**[{s['id']}] {s['citation']}** · relevance {s['score']:.2f}\n> {excerpt}")
    return "\n\n".join(out)


def format_trace(res):
    if not res.get("trace"):
        return "_No trace._"
    lines = [f"{i}. **{t['agent']}**: {t['detail']}" for i, t in enumerate(res["trace"], start=1)]
    if res.get("models_used"):
        lines.append(f"\n**Models used:** {', '.join(res['models_used'])}  ")
    lines.append(f"\n**Latency:** {res.get('latency_s', 0)} s")
    return "\n".join(lines)


def ask(question, chat, history):
    question = (question or "").strip()
    if not question:
        return gr.update(), chat, history, gr.update(), gr.update(), gr.update()
    res = get_assistant().ask(question, history)
    badge = STATUS_BADGE.get(res["status"], "")
    shown = res["answer"] + (f"\n\n<sub>{badge}</sub>" if badge and res["status"] != "chitchat" else "")
    chat = chat + [{"role": "user", "content": question}, {"role": "assistant", "content": shown}]
    if res["status"] in ("ok", "unverified", "not_found", "chitchat"):
        history = history + [(question, res["answer"])]
    return "", chat, history, format_sources(res["sources"]), format_trace(res), badge


def system_status():
    a = get_assistant()
    providers = a.llm.available_providers
    keys = ("🟢 " + " + ".join(providers)) if providers else "🔴 No API key found – add GROQ_API_KEY"
    return (f"**LLM:** {keys} · Reasoner `{settings.reasoner_model}` · Planner/Validator `{settings.planner_model}` · "
            f"Backup `{settings.gemini_models[0]}`  \n**Retrieval:** `{settings.embedding_model}` + BM25 → "
            f"`{getattr(a.store.reranker, 'model_name', 'fused ranking (no re-ranker)')}` · **Indexed chunks:** {a.store.count()}")


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def _chatbot():
    try:  # Gradio 4/5 need type="messages"; Gradio 6 uses messages by default
        return gr.Chatbot(type="messages", height=480, label="Conversation")
    except TypeError:
        return gr.Chatbot(height=480, label="Conversation")


EXAMPLES = [
    "How many days of paternity leave do employees get?",
    "What is the total revenue in the North region?",
    "Which products are below their reorder level?",
    "How long does a refund take, and is UPI faster?",
    "What is the hotel limit in metro cities?",
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Agentic RAG Document Assistant") as demo:
        gr.Markdown("# 🧠 Agentic RAG – Enterprise Document Assistant\n"
                    "Upload PDF, TXT, CSV or Excel files and ask questions. Four AI agents "
                    "(Planner → Retriever → Reasoner → Validator) answer **only from your documents**, with citations.")
        status = gr.Markdown()

        with gr.Tab("1 · Documents"):
            files = gr.File(label="Upload documents", file_count="multiple", type="filepath",
                            file_types=list(settings.allowed_extensions))
            upload_btn = gr.Button("Process & index", variant="primary")
            report = gr.Dataframe(headers=["File", "Status", "Details"], label="Upload report", wrap=True)
            gr.Markdown("### Indexed documents")
            docs = gr.Dataframe(headers=["Document", "Type", "Chunks"], interactive=False)
            with gr.Row():
                to_delete = gr.Dropdown(label="Remove a document", choices=[])
                del_btn = gr.Button("Remove")
                reset_btn = gr.Button("Clear all documents", variant="stop")

        with gr.Tab("2 · Ask"):
            history = gr.State([])
            with gr.Row():
                with gr.Column(scale=3):
                    chat = _chatbot()
                    with gr.Row():
                        q = gr.Textbox(placeholder="Ask a question about your documents…", show_label=False, scale=5)
                        send = gr.Button("Ask", variant="primary", scale=1)
                    gr.Examples(EXAMPLES, inputs=q)
                    clear = gr.Button("Clear conversation")
                with gr.Column(scale=2):
                    badge = gr.Markdown()
                    gr.Markdown("### 📚 Sources")
                    sources = gr.Markdown("_Ask a question to see sources._")
                    with gr.Accordion("🤖 Agent trace", open=False):
                        trace = gr.Markdown()

        with gr.Tab("About"):
            gr.Markdown(Path(__file__).with_name("ABOUT.md").read_text(encoding="utf-8")
                        if Path(__file__).with_name("ABOUT.md").exists() else "See README.md")

        # wiring
        upload_btn.click(upload, files, [report, docs, to_delete]).then(system_status, None, status)
        del_btn.click(delete_doc, to_delete, [docs, to_delete]).then(system_status, None, status)
        reset_btn.click(reset_all, None, [docs, to_delete, chat, history]).then(system_status, None, status)
        for trigger in (send.click, q.submit):
            trigger(ask, [q, chat, history], [q, chat, history, sources, trace, badge]).then(system_status, None, status)
        clear.click(lambda: ([], [], "", "", ""), None, [chat, history, sources, trace, badge])
        demo.load(system_status, None, status).then(docs_table, None, docs).then(doc_choices, None, to_delete)
    return demo


def launch(share: bool | None = None, **kwargs):
    in_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
    on_spaces = "SPACE_ID" in os.environ
    get_assistant()  # load models before the first request
    demo = build_ui()
    demo.queue(default_concurrency_limit=2)
    if on_spaces:
        return demo.launch(server_name="0.0.0.0", **kwargs)
    return demo.launch(share=in_colab if share is None else share, **kwargs)


if __name__ == "__main__":
    launch()
