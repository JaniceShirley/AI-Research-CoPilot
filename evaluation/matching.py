"""Text normalisation and evidence-to-chunk matching.

FAISS chunk IDs are random UUIDs that change every time the index is rebuilt,
so the dataset cannot reference chunks by ID. Instead each relevant source is
described by (paper file, optional 1-based page, verbatim evidence quote). A
retrieved chunk "covers" an evidence item when it comes from the same paper
(and page, if given) and contains enough of the quote's word trigrams.

Trigram coverage (rather than exact substring search) tolerates the small
differences PDF extraction introduces (hyphenation, double spaces, curly
quotes) while still requiring the same word sequence.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Iterable, Optional

_NON_ALNUM = re.compile(r"[^0-9a-z%.]+")


def normalize_text(text: str) -> str:
    """Lowercase, NFKC-normalise and collapse everything that isn't a word."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"\s+%", "%", text)  # "93.96 %" -> "93.96%"
    text = _NON_ALNUM.sub(" ", text)
    # Drop dots that are not decimal points (e.g. sentence ends) so "93.7%."
    # and "93.7%" tokenise the same way.
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalize_text(text).split()


def _ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def evidence_coverage(evidence: str, chunk_text: str, n: int = 3) -> float:
    """Fraction of the evidence's word n-grams that also occur in the chunk."""
    ev = tokens(evidence)
    if not ev:
        return 0.0
    n = min(n, len(ev))
    ev_grams = _ngrams(ev, n)
    chunk_grams = _ngrams(tokens(chunk_text), n)
    if not ev_grams:
        return 0.0
    return len(ev_grams & chunk_grams) / len(ev_grams)


def same_paper(expected: Optional[str], actual: Optional[str]) -> bool:
    """Compare paper file names, ignoring directories and case."""
    if not expected:
        return True
    if not actual:
        return False
    return os.path.basename(expected).lower() == os.path.basename(actual).lower()


def chunk_matches_source(source: dict, chunk: dict, threshold: float = 0.6) -> bool:
    """Does a retrieved chunk cover one relevant-source entry of the dataset?

    ``source``: {"paper": str, "page": int|None (1-based), "evidence": str}
    ``chunk``:  {"source": str, "page": int|None (1-based), "text": str}
    """
    if not same_paper(source.get("paper"), chunk.get("source")):
        return False
    page = source.get("page")
    if page is not None and chunk.get("page") is not None and int(page) != int(chunk["page"]):
        return False
    return evidence_coverage(source.get("evidence", ""), chunk.get("text", "")) >= threshold


def best_coverage(source: dict, chunks: Iterable[dict]) -> float:
    """Highest evidence coverage among chunks from the right paper/page."""
    best = 0.0
    for chunk in chunks:
        if not same_paper(source.get("paper"), chunk.get("source")):
            continue
        page = source.get("page")
        if page is not None and chunk.get("page") is not None and int(page) != int(chunk["page"]):
            continue
        best = max(best, evidence_coverage(source.get("evidence", ""), chunk.get("text", "")))
    return best
