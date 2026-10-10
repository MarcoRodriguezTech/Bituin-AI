"""Ollama-backed RAG chain: retrieve local context, synthesize answer."""
import logging
from datetime import date

import ollama

from config.settings import LLM_MODEL, NUM_CTX, OLLAMA_HOST, RETRIEVE_K
from src.database.vector_store import VectorStore

log = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.client = ollama.Client(host=OLLAMA_HOST)
        self.vector_store = VectorStore()

    def retrieve(self, user_prompt: str, k: int = RETRIEVE_K) -> list[dict]:
        return self.vector_store.search(user_prompt, k=k)

    def _messages(
        self,
        user_prompt: str,
        context_chunks: list[dict],
        history: list[dict] | None = None,
    ) -> list[dict]:
        context_str = "\n\n---\n\n".join(
            f"[{c['metadata']['source']} | {c['metadata']['sender']} | {c['metadata']['subject']} | {c['metadata'].get('timestamp', '')[:10]}]\n{c['content']}"
            for c in context_chunks
        )
        system_prompt = (
            "You are Bituin, a privacy-focused local email assistant. "
            f"Today's date is {date.today().isoformat()}. "
            "Answer using ONLY the email context provided below and the conversation so far. "
            "If the answer cannot be found there, say so clearly. "
            "Follow the format the user asks for (paragraph, bullets, table). "
            "If the user asks to reformat or rewrite a previous answer "
            "(e.g. 'put it in a paragraph', 'make it shorter'), reformat that "
            "answer only — never introduce new emails or information. "
            "Cover every distinct email in the context — be complete but "
            "concise; group near-identical emails into one point.\n\n"
            f"EMAIL CONTEXT:\n{context_str}"
        )
        msgs = [{"role": "system", "content": system_prompt}]
        for m in (history or [])[-6:]:
            msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    def query(self, user_prompt: str, history: list[dict] | None = None) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=self._messages(user_prompt, self.retrieve(user_prompt), history),
            options={"temperature": 0.2, "num_ctx": NUM_CTX},
        )
        return response["message"]["content"]

    def stream_chat(self, user_prompt: str, history: list[dict] | None = None):
        """Conversational reply with no email context — for greetings and
        questions about Bituin itself."""
        system_prompt = (
            "You are Bituin, a privacy-focused local email assistant running "
            f"entirely on the user's own machine. Today's date is {date.today().isoformat()}. "
            "Chat naturally and keep replies short and friendly. "
            "You help with their Gmail inbox: answering questions about specific "
            "emails, summarizing what needs attention, extracting to-dos and "
            "deadlines, and writing a daily briefing. Everything runs on-device "
            "— no email data ever leaves this machine. "
            "You cannot see their emails in this mode; if they ask about "
            "inbox content, invite them to ask the question directly and "
            "you'll search their synced mail."
        )
        msgs = [{"role": "system", "content": system_prompt}]
        for m in (history or [])[-6:]:
            msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": user_prompt})
        for part in self.client.chat(
            model=self.model_name,
            messages=msgs,
            options={"temperature": 0.4, "num_ctx": NUM_CTX},
            stream=True,
        ):
            yield part["message"]["content"]

    def stream_query(
        self,
        user_prompt: str,
        context_chunks: list[dict] | None = None,
        history: list[dict] | None = None,
    ):
        """Token-by-token generator for st.write_stream."""
        chunks = context_chunks if context_chunks is not None else self.retrieve(user_prompt)
        for part in self.client.chat(
            model=self.model_name,
            messages=self._messages(user_prompt, chunks, history),
            options={"temperature": 0.2, "num_ctx": NUM_CTX},
            stream=True,
        ):
            yield part["message"]["content"]

    def is_alive(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:  # noqa: BLE001
            return False
