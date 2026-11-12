"""LLM-as-a-judge scoring for faithfulness and answer relevance.

These scores come from a language model, not from ground truth. They are
labelled ``*_llm_judge`` everywhere they are written out. See README
"Limitations" for why they should be read as estimates.

The judge talks to any object exposing ``chat.completions.create(...)`` in the
OpenAI/Groq style, so tests can pass a fake client.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

FAITHFULNESS_PROMPT = """You are a strict fact-checking judge for a retrieval-augmented QA system.

You get CONTEXT (the only text the assistant was allowed to use) and the assistant's ANSWER.

Steps:
1. Split the ANSWER into atomic factual claims (one checkable fact each). Ignore
   formatting, headings, greetings, and statements that only say information
   is missing or not provided.
2. For each claim decide, using ONLY the CONTEXT (not your own knowledge):
   - "supported": the context states it or it follows directly from the context.
   - "unsupported": the context does not contain enough to confirm it, even if
     it might be true in the real world.
   - "contradicted": the context says something incompatible.
3. For supported/contradicted claims, quote the shortest relevant context span.

Return ONLY a JSON object:
{{"claims": [{{"claim": "...", "verdict": "supported|unsupported|contradicted", "evidence": "..."}}]}}
If the answer makes no factual claims, return {{"claims": []}}.

CONTEXT:
\"\"\"
{context}
\"\"\"

ANSWER:
\"\"\"
{answer}
\"\"\"
"""

RELEVANCE_PROMPT = """You are judging whether an assistant's ANSWER addresses the user's QUESTION.
Judge only relevance and completeness with respect to what was asked, NOT whether
the facts are correct (that is checked separately).

Score with this rubric:
5 = directly and completely answers every part of the question
4 = answers the main question; minor parts missing or some unnecessary padding
3 = partially answers; important parts of the question are missing
2 = mostly off-target; only tangentially related to the question
1 = does not address the question at all (refusal, wrong topic, error message)

Return ONLY a JSON object: {{"score": <integer 1-5>, "reason": "<one sentence>"}}

QUESTION:
\"\"\"
{question}
\"\"\"

ANSWER:
\"\"\"
{answer}
\"\"\"
"""

VERDICTS = ("supported", "unsupported", "contradicted")


class JudgeError(RuntimeError):
    pass


def parse_json_object(text: str) -> dict:
    """Parse the first JSON object in ``text`` (tolerates code fences / chatter)."""
    if not text:
        raise JudgeError("empty judge response")
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise JudgeError(f"no JSON object in judge response: {text[:200]!r}")
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise JudgeError(f"invalid JSON from judge: {e}") from e
    if not isinstance(obj, dict):
        raise JudgeError("judge response is not a JSON object")
    return obj


def score_faithfulness(parsed: dict) -> dict:
    """Turn the judge's claim list into a score.

    faithfulness = supported claims / all claims. Unsupported and contradicted
    claims both count against the answer: a claim the context does not back up
    is never treated as factual. With zero claims the score is ``None``.
    """
    claims = parsed.get("claims")
    if not isinstance(claims, list):
        raise JudgeError("judge JSON has no 'claims' list")
    clean = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        verdict = str(c.get("verdict", "")).strip().lower()
        if verdict not in VERDICTS:
            verdict = "unsupported"  # unknown labels are not given the benefit of the doubt
        clean.append({
            "claim": str(c.get("claim", "")).strip(),
            "verdict": verdict,
            "evidence": str(c.get("evidence", "") or "").strip(),
        })
    counts = {v: sum(1 for c in clean if c["verdict"] == v) for v in VERDICTS}
    total = len(clean)
    return {
        "score": (counts["supported"] / total) if total else None,
        "n_claims": total,
        "n_supported": counts["supported"],
        "n_unsupported": counts["unsupported"],
        "n_contradicted": counts["contradicted"],
        "claims": clean,
    }


def score_relevance(parsed: dict) -> dict:
    raw = parsed.get("score")
    try:
        raw = int(round(float(raw)))
    except (TypeError, ValueError):
        raise JudgeError(f"judge returned non-numeric score: {raw!r}")
    if not 1 <= raw <= 5:
        raise JudgeError(f"judge score out of range 1-5: {raw}")
    return {
        "rubric_score": raw,
        "score": (raw - 1) / 4.0,  # normalised to 0..1
        "reason": str(parsed.get("reason", "")).strip(),
    }


def retry_wait_seconds(error_message: str, attempt: int, backoff_s: float) -> float:
    """How long to wait before retrying a failed judge call.

    Rate-limit errors (Groq limits are per minute) wait for the time the API
    suggests, or 60 s; other errors use a short linear backoff.
    """
    msg = error_message.lower()
    if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
        m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
        if m:
            return float(m.group(1) or 0) * 60 + float(m.group(2)) + 1
        return 60.0
    return backoff_s * (attempt + 1)


class LLMJudge:
    """Thin wrapper around a chat-completions client with retries."""

    def __init__(self, client: Any, model: str, max_context_chars: int = 24000,
                 retries: int = 4, backoff_s: float = 5.0, max_tokens: Optional[int] = None):
        self.client = client
        self.model = model
        self.max_context_chars = max_context_chars
        self.retries = retries
        self.backoff_s = backoff_s
        self.max_tokens = max_tokens
        self._json_mode = True

    def _complete(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.retries):
            kwargs = dict(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            if self._json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:  # network, rate limit, unsupported param
                last_err = e
                msg = str(e).lower()
                if self._json_mode and "response_format" in msg:
                    self._json_mode = False  # model doesn't support JSON mode; retry plain
                    continue
                if attempt < self.retries - 1 and self.backoff_s > 0:
                    time.sleep(retry_wait_seconds(str(e), attempt, self.backoff_s))
        raise JudgeError(f"judge call failed after {self.retries} attempts: {last_err}")

    def faithfulness(self, context: str, answer: str) -> dict:
        ctx = context[: self.max_context_chars]
        out = score_faithfulness(parse_json_object(
            self._complete(FAITHFULNESS_PROMPT.format(context=ctx, answer=answer))))
        out["context_truncated"] = len(context) > self.max_context_chars
        return out

    def relevance(self, question: str, answer: str) -> dict:
        return score_relevance(parse_json_object(
            self._complete(RELEVANCE_PROMPT.format(question=question, answer=answer))))
