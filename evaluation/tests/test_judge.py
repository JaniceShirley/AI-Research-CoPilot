import json
import unittest
from types import SimpleNamespace

from evaluation.judge import (JudgeError, LLMJudge, parse_json_object, retry_wait_seconds,
                              score_faithfulness, score_relevance)


def fake_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeClient:
    """Mimics groq.Groq().chat.completions.create."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return fake_response(r)


class ParseTests(unittest.TestCase):
    def test_plain_fenced_and_chatty(self):
        self.assertEqual(parse_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json_object('```json\n{"a": 2}\n```'), {"a": 2})
        self.assertEqual(parse_json_object('Sure! {"a": 3} hope that helps'), {"a": 3})

    def test_garbage_raises(self):
        for bad in ("", "no json here", "[1, 2]", "{broken"):
            with self.assertRaises(JudgeError):
                parse_json_object(bad)


class FaithfulnessScoringTests(unittest.TestCase):
    def test_unsupported_and_contradicted_count_against(self):
        out = score_faithfulness({"claims": [
            {"claim": "a", "verdict": "supported"},
            {"claim": "b", "verdict": "supported"},
            {"claim": "c", "verdict": "unsupported"},
            {"claim": "d", "verdict": "contradicted"},
        ]})
        self.assertEqual(out["score"], 0.5)
        self.assertEqual((out["n_supported"], out["n_unsupported"], out["n_contradicted"]), (2, 1, 1))

    def test_unknown_verdict_treated_as_unsupported(self):
        out = score_faithfulness({"claims": [{"claim": "x", "verdict": "probably true"}]})
        self.assertEqual(out["score"], 0.0)
        self.assertEqual(out["claims"][0]["verdict"], "unsupported")

    def test_no_claims_is_undefined(self):
        self.assertIsNone(score_faithfulness({"claims": []})["score"])

    def test_missing_claims_key(self):
        with self.assertRaises(JudgeError):
            score_faithfulness({"score": 1})


class RelevanceScoringTests(unittest.TestCase):
    def test_normalisation(self):
        self.assertEqual(score_relevance({"score": 5})["score"], 1.0)
        self.assertEqual(score_relevance({"score": 1})["score"], 0.0)
        self.assertEqual(score_relevance({"score": "3"})["score"], 0.5)

    def test_out_of_range_or_non_numeric(self):
        for bad in (0, 6, "high", None):
            with self.assertRaises(JudgeError):
                score_relevance({"score": bad})


class LLMJudgeTests(unittest.TestCase):
    def test_end_to_end_with_fake_client(self):
        client = FakeClient([
            json.dumps({"claims": [{"claim": "uses XGBoost", "verdict": "supported", "evidence": "XGBoost"}]}),
            json.dumps({"score": 4, "reason": "mostly answers"}),
        ])
        j = LLMJudge(client, "judge-model", backoff_s=0)
        f = j.faithfulness("context mentions XGBoost", "It uses XGBoost.")
        r = j.relevance("Which algorithm?", "It uses XGBoost.")
        self.assertEqual(f["score"], 1.0)
        self.assertFalse(f["context_truncated"])
        self.assertEqual(r["rubric_score"], 4)
        self.assertEqual(client.calls[0]["model"], "judge-model")
        self.assertEqual(client.calls[0]["temperature"], 0)
        self.assertIn("context mentions XGBoost", client.calls[0]["messages"][0]["content"])

    def test_retries_then_fails(self):
        client = FakeClient([RuntimeError("429 rate limit")] * 3)
        j = LLMJudge(client, "m", retries=3, backoff_s=0)
        with self.assertRaises(JudgeError):
            j.relevance("q", "a")
        self.assertEqual(len(client.calls), 3)

    def test_falls_back_when_json_mode_unsupported(self):
        client = FakeClient([RuntimeError("response_format not supported"), '{"score": 5}'])
        j = LLMJudge(client, "m", backoff_s=0)
        self.assertEqual(j.relevance("q", "a")["rubric_score"], 5)
        self.assertIn("response_format", client.calls[0])
        self.assertNotIn("response_format", client.calls[1])

    def test_retry_wait(self):
        self.assertEqual(retry_wait_seconds("Error code: 429 ... Please try again in 7.5s.", 0, 5), 8.5)
        self.assertEqual(retry_wait_seconds("429 rate limit; try again in 1m2.0s", 0, 5), 63.0)
        self.assertEqual(retry_wait_seconds("Error code: 429 rate_limit_exceeded", 0, 5), 60.0)
        self.assertEqual(retry_wait_seconds("connection reset", 1, 5), 10)

    def test_max_tokens_passed(self):
        client = FakeClient(['{"score": 3}'])
        LLMJudge(client, "m", max_tokens=900, backoff_s=0).relevance("q", "a")
        self.assertEqual(client.calls[0]["max_tokens"], 900)

    def test_context_truncation_flag(self):
        client = FakeClient(['{"claims": []}'])
        j = LLMJudge(client, "m", max_context_chars=10, backoff_s=0)
        self.assertTrue(j.faithfulness("x" * 50, "answer")["context_truncated"])


if __name__ == "__main__":
    unittest.main()
