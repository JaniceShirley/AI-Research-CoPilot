import json
import os
import tempfile
import unittest

from evaluation.dataset import (DatasetError, load_dataset, resolve_paper_files, select_items,
                                validate_dataset)

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(os.path.dirname(HERE), "datasets")


def minimal():
    return {
        "papers": [{"file": "a.pdf"}],
        "items": [{"id": "q1", "question": "What?", "expected_answer": "That.",
                   "relevant_sources": [{"paper": "a.pdf", "page": 1, "evidence": "quote"}]}],
    }


class DatasetTests(unittest.TestCase):
    def test_minimal_is_valid(self):
        validate_dataset(minimal())

    def test_shipped_datasets_are_valid(self):
        for name in ("parkease_papers.json", "template.json"):
            data = load_dataset(os.path.join(DATASETS, name))
            self.assertTrue(data["items"])
            self.assertEqual(len(data["_sha256"]), 64)

    def test_real_dataset_items_are_marked_unverified_until_reviewed(self):
        data = load_dataset(os.path.join(DATASETS, "parkease_papers.json"))
        # Guards against silently presenting the Claude-drafted dataset as reviewed.
        if data.get("status") == "verified":
            self.assertTrue(all(it.get("verified") is True for it in data["items"]))

    def test_reports_all_problems(self):
        d = minimal()
        d["items"].append({"id": "q1", "question": " ", "expected_answer": "",
                           "relevant_sources": [{"paper": "missing.pdf", "page": 0}]})
        with self.assertRaises(DatasetError) as cm:
            validate_dataset(d)
        msg = str(cm.exception)
        for fragment in ("duplicate id", "empty question", "not listed in 'papers'",
                         "missing 'evidence'", "1-based integer"):
            self.assertIn(fragment, msg)

    def test_resolve_and_select(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(DatasetError):
                resolve_paper_files(minimal(), d)
            open(os.path.join(d, "a.pdf"), "wb").close()
            self.assertEqual(resolve_paper_files(minimal(), d), [os.path.join(d, "a.pdf")])
        data = minimal()
        self.assertEqual(len(select_items(data, ["q1"])), 1)
        with self.assertRaises(DatasetError):
            select_items(data, ["nope"])

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{not json")
        try:
            with self.assertRaises(DatasetError):
                load_dataset(f.name)
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
