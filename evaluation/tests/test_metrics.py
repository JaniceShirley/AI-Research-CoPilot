import unittest

from evaluation import metrics
from evaluation.matching import evidence_coverage, normalize_text, chunk_matches_source

PAPER = "paper.pdf"


def chunk(text, page=1, source=PAPER):
    return {"source": source, "page": page, "text": text}


class MatchingTests(unittest.TestCase):
    def test_normalize_handles_pdf_noise(self):
        self.assertEqual(normalize_text("Real -time  LPR, 93.7%."), "real time lpr 93.7%")

    def test_coverage_full_and_partial(self):
        ev = "the overall rate of success is 93.7%"
        self.assertEqual(evidence_coverage(ev, "Combining, the overall rate of success is 93.7%."), 1.0)
        self.assertEqual(evidence_coverage(ev, "completely unrelated text about parking"), 0.0)
        partial = evidence_coverage(ev, "the overall rate of success was high")
        self.assertTrue(0 < partial < 1)

    def test_paper_and_page_must_match(self):
        src = {"paper": PAPER, "page": 2, "evidence": "four distinct colors"}
        self.assertTrue(chunk_matches_source(src, chunk("only four distinct colors are used", page=2)))
        self.assertFalse(chunk_matches_source(src, chunk("only four distinct colors are used", page=3)))
        self.assertFalse(chunk_matches_source(src, chunk("only four distinct colors are used", page=2,
                                                         source="other.pdf")))
        # page omitted in dataset -> any page of the right paper is fine
        self.assertTrue(chunk_matches_source({"paper": PAPER, "evidence": "four distinct colors"},
                                             chunk("four distinct colors", page=9)))


class RecallTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"paper": PAPER, "page": 1, "evidence": "alpha beta gamma delta"},
            {"paper": PAPER, "page": 2, "evidence": "epsilon zeta eta theta"},
        ]
        self.ranked = [
            chunk("noise one two three"),
            chunk("text alpha beta gamma delta text", page=1),
            chunk("noise four five six"),
            chunk("epsilon zeta eta theta", page=2),
        ]

    def test_recall_at_k_values(self):
        self.assertEqual(metrics.recall_at_k(self.sources, self.ranked, 1), 0.0)
        self.assertEqual(metrics.recall_at_k(self.sources, self.ranked, 2), 0.5)
        self.assertEqual(metrics.recall_at_k(self.sources, self.ranked, 4), 1.0)
        self.assertEqual(metrics.recall_at_k(self.sources, self.ranked, 100), 1.0)

    def test_hit_and_rank(self):
        self.assertEqual(metrics.hit_at_k(self.sources, self.ranked, 1), 0.0)
        self.assertEqual(metrics.hit_at_k(self.sources, self.ranked, 2), 1.0)
        self.assertEqual(metrics.first_relevant_rank(self.sources, self.ranked), 2)
        self.assertIsNone(metrics.first_relevant_rank(self.sources, []))

    def test_no_relevant_sources_is_undefined_not_zero(self):
        self.assertIsNone(metrics.recall_at_k([], self.ranked, 5))

    def test_empty_retrieval_is_zero(self):
        self.assertEqual(metrics.recall_at_k(self.sources, [], 5), 0.0)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            metrics.recall_at_k(self.sources, self.ranked, 0)


class SupplementaryMetricTests(unittest.TestCase):
    def test_unsupported_numbers(self):
        ctx = "The model achieved precision of 96.35% and 9,551 images."
        ans = "Precision was 96.35% on 9551 images with recall 77.10%. See step 2."
        self.assertEqual(metrics.unsupported_numbers(ans, ctx), ["77.10%"])

    def test_lexical_support(self):
        self.assertEqual(metrics.lexical_support("xgboost predicts prices", "XGBoost predicts prices well"), 1.0)
        self.assertEqual(metrics.lexical_support("banana orange", "xgboost"), 0.0)
        self.assertIsNone(metrics.lexical_support("the and of", "anything"))

    def test_key_fact_coverage(self):
        ans = "It reached 93.96 % mAP50 using YOLOv12."
        self.assertEqual(metrics.key_fact_coverage(ans, ["93.96%", "YOLOv12"]), 1.0)
        self.assertEqual(metrics.key_fact_coverage(ans, ["93.96%", "Tesseract"]), 0.5)
        self.assertIsNone(metrics.key_fact_coverage(ans, []))
        # whole-token match: "9" must not match inside "93.96"
        self.assertEqual(metrics.key_fact_coverage(ans, ["9"]), 0.0)

    def test_cosine(self):
        self.assertAlmostEqual(metrics.cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(metrics.cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertIsNone(metrics.cosine_similarity([0, 0], [1, 1]))
        self.assertIsNone(metrics.cosine_similarity([1], [1, 2]))

    def test_summarize_ignores_none(self):
        s = metrics.summarize([1.0, None, 3.0, 2.0])
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["mean"], 2.0)
        self.assertEqual(s["median"], 2.0)
        self.assertAlmostEqual(s["p95"], 2.9)
        self.assertEqual(metrics.summarize([None])["n"], 0)
        self.assertIsNone(metrics.summarize([])["mean"])


if __name__ == "__main__":
    unittest.main()
