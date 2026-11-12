"""Observe the production pipeline from the outside.

``ask_question`` returns only the answer string. To measure retrieval and
generation separately without editing ``backend/rag_pipeline.py``, this module
temporarily wraps:

* the FAISS search methods the pipeline uses (``similarity_search_with_score``,
  ``similarity_search``, ``max_marginal_relevance_search``) to record the
  chunks returned, in order, and the time spent (query embedding + search);
* ``rag_pipeline.client`` with a proxy whose ``chat.completions.create`` times
  every Groq call and keeps the prompt and response;
* stdout, to read the pipeline's own ``Query route: ...`` log lines.

All patches are undone when the context manager exits, even on error.
"""

from __future__ import annotations

import contextlib
import io
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

RETRIEVAL_METHODS = (
    "similarity_search_with_score",
    "similarity_search",
    "max_marginal_relevance_search",
)


def doc_to_chunk(doc: Any) -> dict:
    """Convert a LangChain Document into a plain dict (page is 1-based)."""
    meta = dict(getattr(doc, "metadata", {}) or {})
    page = meta.get("page")
    try:
        page = int(page) + 1 if page is not None else None
    except (TypeError, ValueError):
        page = None
    return {
        "source": meta.get("source"),
        "paper_id": meta.get("paper_id"),
        "page": page,
        "text": getattr(doc, "page_content", "") or "",
    }


@dataclass
class RetrievalCall:
    method: str
    query: str
    kwargs: dict
    seconds: float
    chunks: list = field(default_factory=list)
    scores: Optional[list] = None


@dataclass
class LLMCall:
    model: str
    seconds: float
    prompt: str
    response: str
    error: Optional[str] = None


@dataclass
class Trace:
    retrieval_calls: list = field(default_factory=list)
    llm_calls: list = field(default_factory=list)
    stdout: str = ""

    @property
    def retrieval_seconds(self) -> float:
        return sum(c.seconds for c in self.retrieval_calls)

    @property
    def generation_seconds(self) -> float:
        return sum(c.seconds for c in self.llm_calls)

    def ranked_chunks(self) -> list:
        """Every chunk FAISS returned, in the order returned, de-duplicated."""
        seen, out = set(), []
        for call in self.retrieval_calls:
            for ch in call.chunks:
                key = (ch["source"], ch["page"], ch["text"])
                if key not in seen:
                    seen.add(key)
                    out.append(ch)
        return out

    def context_chunks(self) -> list:
        """Retrieved chunks that were actually placed in the final LLM prompt,
        ordered by where they appear in the prompt."""
        if not self.llm_calls:
            return []
        prompt = self.llm_calls[-1].prompt
        located = []
        for ch in self.ranked_chunks():
            text = ch["text"]
            pos = prompt.find(text) if text else -1
            if pos >= 0:
                located.append((pos, ch))
        located.sort(key=lambda x: x[0])
        return [ch for _, ch in located]

    def answer_context(self) -> str:
        """The context block of the final prompt (what the answer may rely on)."""
        if not self.llm_calls:
            return ""
        return extract_context(self.llm_calls[-1].prompt)

    def routes(self) -> list:
        return re.findall(r"Query route(?: after gate)?: ([A-Z_]+)", self.stdout)


_CONTEXT_PATTERNS = [
    re.compile(r"Research Context:\s*(.*?)\n\s*User Question:", re.S),
    re.compile(r"\nContext:\s*(.*?)\n\s*Question:", re.S),
]


def extract_context(prompt: str) -> str:
    """Pull the context section out of one of the pipeline's prompt templates.
    Falls back to the whole prompt if the template is not recognised."""
    for pat in _CONTEXT_PATTERNS:
        m = pat.search(prompt or "")
        if m:
            return m.group(1).strip()
    return prompt or ""


class _CompletionsProxy:
    def __init__(self, real_completions: Any, trace_getter):
        self._real = real_completions
        self._trace_getter = trace_getter

    def create(self, *args, **kwargs):
        messages = kwargs.get("messages") or []
        prompt = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        start = time.perf_counter()
        try:
            resp = self._real.create(*args, **kwargs)
        except Exception as e:
            trace = self._trace_getter()
            if trace is not None:
                trace.llm_calls.append(LLMCall(kwargs.get("model", ""), time.perf_counter() - start,
                                               prompt, "", error=str(e)))
            raise
        elapsed = time.perf_counter() - start
        trace = self._trace_getter()
        if trace is not None:
            try:
                text = resp.choices[0].message.content or ""
            except Exception:
                text = ""
            trace.llm_calls.append(LLMCall(kwargs.get("model", ""), elapsed, prompt, text))
        return resp

    def __getattr__(self, name):
        return getattr(self._real, name)


class _ChatProxy:
    def __init__(self, real_chat: Any, trace_getter):
        self._real = real_chat
        self.completions = _CompletionsProxy(real_chat.completions, trace_getter)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _ClientProxy:
    def __init__(self, real_client: Any, trace_getter):
        self._real = real_client
        self.chat = _ChatProxy(real_client.chat, trace_getter)

    def __getattr__(self, name):
        return getattr(self._real, name)


class PipelineInstrumentation:
    """Install once around a whole evaluation run; call ``capture()`` per question.

    ``pipeline``: the imported ``rag_pipeline`` module (or a test double with a
    ``client`` attribute).
    ``vectorstore_cls``: the class whose search methods to wrap (FAISS).
    """

    def __init__(self, pipeline: Any, vectorstore_cls: Any):
        self.pipeline = pipeline
        self.vectorstore_cls = vectorstore_cls
        self._local = threading.local()
        self._originals: dict = {}
        self._orig_client = None

    def _trace(self) -> Optional[Trace]:
        return getattr(self._local, "trace", None)

    def _wrap_search(self, name: str, original):
        instr = self

        def wrapper(vs_self, query, *args, **kwargs):
            depth = getattr(instr._local, "depth", 0)
            instr._local.depth = depth + 1
            start = time.perf_counter()
            try:
                result = original(vs_self, query, *args, **kwargs)
            finally:
                instr._local.depth = depth
            elapsed = time.perf_counter() - start
            trace = instr._trace()
            # Only record the outermost call (similarity_search calls
            # similarity_search_with_score internally).
            if trace is not None and depth == 0:
                if name == "similarity_search_with_score":
                    docs = [d for d, _ in result]
                    scores = [float(s) for _, s in result]
                else:
                    docs, scores = list(result), None
                call_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, (int, float, str))}
                if args:
                    call_kwargs["k"] = args[0]
                trace.retrieval_calls.append(RetrievalCall(
                    name, query, call_kwargs, elapsed, [doc_to_chunk(d) for d in docs], scores))
            return result

        wrapper.__wrapped__ = original
        return wrapper

    def __enter__(self):
        for name in RETRIEVAL_METHODS:
            original = getattr(self.vectorstore_cls, name, None)
            if original is None:
                continue
            self._originals[name] = original
            setattr(self.vectorstore_cls, name, self._wrap_search(name, original))
        self._orig_client = self.pipeline.client
        self.pipeline.client = _ClientProxy(self._orig_client, self._trace)
        return self

    def __exit__(self, *exc):
        for name, original in self._originals.items():
            setattr(self.vectorstore_cls, name, original)
        self._originals.clear()
        if self._orig_client is not None:
            self.pipeline.client = self._orig_client
            self._orig_client = None
        return False

    @contextlib.contextmanager
    def capture(self, echo: bool = False):
        """Collect a ``Trace`` for everything the pipeline does inside the block."""
        trace = Trace()
        self._local.trace = trace
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                yield trace
        finally:
            self._local.trace = None
            trace.stdout = buf.getvalue()
            if echo and trace.stdout:
                print(trace.stdout, end="")
