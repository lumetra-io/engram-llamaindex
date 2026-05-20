"""EngramMemory — a `BaseMemory` implementation backed by the Engram REST API.

Engram (https://lumetra.io) is Lumetra's durable, explainable memory service for
AI agents. This module exposes `EngramMemory`, a drop-in replacement for
LlamaIndex's built-in chat-history memory classes that persists every message
to an Engram bucket and can perform semantic retrieval against it.

Typical usage::

    from llama_index.memory.engram import EngramMemory
    from llama_index.llms.openai import OpenAI
    from llama_index.core.agent.workflow import FunctionAgent

    memory = EngramMemory.from_defaults(
        bucket="user-42",          # one bucket per user / session / agent
        api_key="eng_live_...",    # or set ENGRAM_API_KEY in the env
    )

    agent = FunctionAgent(llm=OpenAI("gpt-4o"), tools=[...])
    response = await agent.run("What did we decide last time?", memory=memory)

Why this is a `BaseMemory` and not a `BaseMemoryBlock`:

* `BaseMemory` is the top-level chat-history interface an agent talks to via
  `put` / `get` / `get_all` / `set` / `reset`. Implementing it lets Engram
  *replace* short-term chat memory entirely — every message round-trips to
  the Engram REST API, and `get(input=...)` can run a semantic query so
  retrieval isn't bounded by a FIFO token window.
* `BaseMemoryBlock` is a sub-component used *inside* a `Memory` object for
  extracting long-term facts from messages that have been flushed out of
  short-term history. It does not own the chat-history contract; agents
  don't call it directly.

Either model is reasonable; we picked `BaseMemory` because Engram's
retrieval pipeline (BM25 + vector + graph + reranker) replaces the entire
short-term/long-term split rather than augmenting it.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.memory import BaseMemory

DEFAULT_BASE_URL = "https://api.lumetra.io"
DEFAULT_BUCKET = "default"
DEFAULT_READ_LIMIT = 50
DEFAULT_TIMEOUT = 120.0  # /v1/query runs an LLM synthesis pass; 30s is too tight.

# Encoded prefix so we can round-trip role information through a flat
# Engram "content" string. Kept short and human-readable on purpose —
# Engram's extractor sees these too, so we want them to read naturally.
_ROLE_PREFIXES: Dict[MessageRole, str] = {
    MessageRole.USER: "USER",
    MessageRole.ASSISTANT: "ASSISTANT",
    MessageRole.SYSTEM: "SYSTEM",
    MessageRole.TOOL: "TOOL",
    MessageRole.FUNCTION: "FUNCTION",
    MessageRole.CHATBOT: "ASSISTANT",
    MessageRole.MODEL: "ASSISTANT",
}
_ROLE_FROM_PREFIX: Dict[str, MessageRole] = {
    "USER": MessageRole.USER,
    "ASSISTANT": MessageRole.ASSISTANT,
    "SYSTEM": MessageRole.SYSTEM,
    "TOOL": MessageRole.TOOL,
    "FUNCTION": MessageRole.FUNCTION,
}


def _encode(message: ChatMessage) -> str:
    role = _ROLE_PREFIXES.get(message.role, "USER")
    content = message.content if isinstance(message.content, str) else str(message.content or "")
    return f"{role}: {content}"


def _decode(content: str) -> ChatMessage:
    if ": " in content:
        prefix, body = content.split(": ", 1)
        role = _ROLE_FROM_PREFIX.get(prefix.strip().upper())
        if role is not None:
            return ChatMessage(role=role, content=body)
    return ChatMessage(role=MessageRole.USER, content=content)


class EngramMemory(BaseMemory):
    """Chat memory backed by an Engram bucket.

    Every message passed to `put` becomes one memory in the configured
    bucket. `get` returns the most recent `read_limit` messages decoded
    back into LlamaIndex `ChatMessage` objects, in chronological order
    (oldest -> newest, matching what agents expect when replaying history).

    For semantic recall over the *entire* bucket — not just the most recent
    window — call `query(question)` directly. This hits Engram's hybrid
    retrieval (BM25 + vector + knowledge graph + reranker) and returns the
    raw REST response.

    Parameters
    ----------
    bucket:
        Engram bucket to scope this memory to. Different agents / users /
        sessions should use different buckets.
    api_key:
        Engram API key. Falls back to the ``ENGRAM_API_KEY`` env var.
    base_url:
        Engram REST base URL (override for self-hosted deployments).
    read_limit:
        How many recent memories `get` / `get_all` return.
    timeout:
        Per-request HTTP timeout in seconds.
    """

    bucket: str = Field(default=DEFAULT_BUCKET)
    base_url: str = Field(default=DEFAULT_BASE_URL)
    read_limit: int = Field(default=DEFAULT_READ_LIMIT)
    timeout: float = Field(default=DEFAULT_TIMEOUT)

    _api_key: str = PrivateAttr()
    _session: requests.Session = PrivateAttr()

    def __init__(
        self,
        bucket: str = DEFAULT_BUCKET,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        read_limit: int = DEFAULT_READ_LIMIT,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            bucket=bucket,
            base_url=base_url.rstrip("/"),
            read_limit=read_limit,
            timeout=timeout,
            **kwargs,
        )
        resolved = api_key or os.environ.get("ENGRAM_API_KEY")
        if not resolved:
            raise ValueError(
                "EngramMemory requires an API key. Pass `api_key=...` or set "
                "the ENGRAM_API_KEY environment variable."
            )
        self._api_key = resolved
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "llama-index-memory-engram/0.1.0",
            }
        )

    # ------------------------------------------------------------------ #
    # LlamaIndex constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def class_name(cls) -> str:
        return "EngramMemory"

    @classmethod
    def from_defaults(
        cls,
        bucket: str = DEFAULT_BUCKET,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        read_limit: int = DEFAULT_READ_LIMIT,
        timeout: float = DEFAULT_TIMEOUT,
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> "EngramMemory":
        """Create an `EngramMemory`, optionally seeding it with `chat_history`."""
        # Strip kwargs LlamaIndex callers sometimes pass through (`llm=`,
        # `token_limit=`, ...) that aren't meaningful for Engram. We
        # silently ignore them so the class stays a drop-in replacement
        # for `ChatMemoryBuffer.from_defaults`.
        kwargs.pop("llm", None)
        kwargs.pop("token_limit", None)
        kwargs.pop("tokenizer_fn", None)
        kwargs.pop("chat_store", None)
        kwargs.pop("chat_store_key", None)

        memory = cls(
            bucket=bucket,
            api_key=api_key,
            base_url=base_url,
            read_limit=read_limit,
            timeout=timeout,
            **kwargs,
        )
        if chat_history:
            memory.put_messages(chat_history)
        return memory

    # ------------------------------------------------------------------ #
    # BaseMemory abstract methods
    # ------------------------------------------------------------------ #

    def put(self, message: ChatMessage) -> None:
        body = _encode(message)
        if not body.split(": ", 1)[-1].strip():
            return  # don't persist empty messages
        self._post(f"/v1/buckets/{self.bucket}/memories", {"content": body})

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        # NOTE: `input` is currently used only as a hint for future
        # retrieval modes. Returning the recency window matches what
        # agents expect from chat-history `get`, and Engram's hybrid
        # retrieval lives on `query()` for users who want it.
        return self.get_all()

    def get_all(self) -> List[ChatMessage]:
        data = self._get(
            f"/v1/buckets/{self.bucket}/memories",
            params={"limit": self.read_limit},
        )
        memories = data.get("memories", []) if isinstance(data, dict) else []
        # Engram returns newest-first; LlamaIndex expects chronological order.
        return [_decode(m.get("content", "")) for m in reversed(memories)]

    def put_messages(self, messages: List[ChatMessage]) -> None:
        for m in messages:
            self.put(m)

    def set(self, messages: List[ChatMessage]) -> None:
        self.reset()
        self.put_messages(messages)

    def reset(self) -> None:
        self._delete(f"/v1/buckets/{self.bucket}/memories")

    # ------------------------------------------------------------------ #
    # Engram-specific extensions
    # ------------------------------------------------------------------ #

    def query(self, question: str) -> Dict[str, Any]:
        """Run Engram's hybrid retrieval over the bucket and return the
        raw REST response (``{success, answer, memories_found, ...}``).

        Use this when you want semantic recall over the entire history
        rather than the last `read_limit` messages.
        """
        return self._post("/v1/query", {"query": question, "bucket": self.bucket})

    def list_buckets(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List Engram buckets visible to this API key."""
        return self._get("/v1/buckets", params={"limit": limit, "offset": offset})

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a single memory by id from the configured bucket."""
        return self._delete(f"/v1/buckets/{self.bucket}/memories/{memory_id}")

    # ------------------------------------------------------------------ #
    # HTTP helpers
    # ------------------------------------------------------------------ #

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._session.post(self._url(path), json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json() if r.content else {}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        r = self._session.get(self._url(path), params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json() if r.content else {}

    def _delete(self, path: str) -> Dict[str, Any]:
        r = self._session.delete(self._url(path), timeout=self.timeout)
        r.raise_for_status()
        return r.json() if r.content else {}
