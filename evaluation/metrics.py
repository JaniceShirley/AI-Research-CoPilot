"""Deterministic evaluation metrics (no LLM involved).

Every function here is pure so it can be unit-tested without FAISS, Groq or
any PDFs. LLM-as-a-judge scores live in ``judge.py``.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Iterable, Optional, Sequence

from .matching import chunk_matches_source, normalize_text, tokens

# ---------------------------------------------------------------------------
# A. Retrieval Recall@K
# ---------------------------------------------------------------------------


def recall_at_k(
    relevant_sources: Sequence[dict],
    ranked_chunks: Sequence[dict],
    k: int,
    threshold: float = 0.6,
) -> Optional[float]:
    """Fraction of relevant evidence items covered by the top-``k`` chunks.

    Returns ``None`` when the question has no relevant sources (recall is
    undefined), so it is excluded from averages instead of counting as 0 or 1.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_sources:
        return None
    top = list(ranked_chunks)[:k]
    found = sum(
        1 for src in relevant_sources
        if any(chunk_matches_source(src, ch, threshold) for ch in top)
    )
    return found / len(relevant_sources)


def hit_at_k(relevant_sources, ranked_chunks, k, threshold=0.6) -> Optional[float]:
    """1.0 if at least one relevant evidence item is in the top-``k``."""
    r = recall_at_k(relevant_sources, ranked_chunks, k, threshold)
    if r is None:
        return None
    return 1.0 if r > 0 else 0.0


def first_relevant_rank(relevant_sources, ranked_chunks, threshold=0.6) -> Optional[int]:
    """1-based rank of the first chunk covering any relevant evidence."""
    for i, ch in enumerate(ranked_chunks, start=1):
        if any(chunk_matches_source(src, ch, threshold) for src in relevant_sources):
            return i
    return None


# ---------------------------------------------------------------------------
# Supplementary, transparent text checks
# ---------------------------------------------------------------------------

_STOPWORDS = set("""
a an and are as at be been being but by can could did do does for from had has
have how if in into is it its may might more most no not of on or our over such
than that the their them then there these they this those to was were what when
where which while who why will with would you your also based using used use
paper papers source section note specific details provided context system
""".split())

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*%?")


def _content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if len(t) > 2 and t not in _STOPWORDS]


def lexical_support(answer: str, context: str) -> Optional[float]:
    """Share of the answer's content words that also appear in the context.

    A crude, fully transparent proxy. High overlap does NOT prove an answer is
    faithful (it can recombine context words into a false claim) and low
    overlap can come from paraphrasing. Reported only as a supplement to the
    claim-level faithfulness judge.
    """
    ans = _content_tokens(answer)
    if not ans:
        return None
    ctx = set(tokens(context))
    return sum(1 for t in ans if t in ctx) / len(ans)


def _norm_number(n: str) -> str:
    return n.replace(",", "").rstrip("%").rstrip(".")


def unsupported_numbers(answer: str, context: str) -> list[str]:
    """Numbers stated in the answer that never appear in the context.

    Numbers are the easiest hallucinations to check mechanically (accuracy
    figures, dataset sizes). Small integers 0-10 are ignored because they are
    usually list numbering or section references.
    """
    ctx_nums = {_norm_number(n) for n in _NUMBER.findall(context or "")}
    out = []
    for n in _NUMBER.findall(answer or ""):
        key = _norm_number(n)
        try:
            if float(key) <= 10 and "." not in key:
                continue
        except ValueError:
            continue
        if key not in ctx_nums and n not in out:
            out.append(n)
    return out


def key_fact_coverage(answer: str, key_facts: Sequence[str]) -> Optional[float]:
    """Fraction of the dataset's key facts that appear in the answer text.

    Key facts are short strings (a number, a model name) taken from the paper.
    Matching is on normalised text, so "93.96 %" matches "93.96%".
    """
    if not key_facts:
        return None
    ans = " " + normalize_text(answer) + " "
    hits = sum(1 for f in key_facts if " " + normalize_text(f) + " " in ans)
    return hits / len(key_facts)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolated percentile (same as numpy's default)."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * pct / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def summarize(values: Iterable[Optional[float]]) -> dict:
    """Mean/median/p95/min/max over the non-None values, plus counts."""
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    s = sorted(vals)
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p95": _percentile(s, 95),
        "min": s[0],
        "max": s[-1],
    }
