"""Runner, instrumentation and data-isolation tests (no network, no models)."""

import json
import os
import shutil
import tempfile
import unittest

from evaluation.instrumentation import PipelineInstrumentation, extract_context
from evaluation.report import write_all
from evaluation.runner import RunConfig, run_evaluation
from evaluation.workspace import (diff_snapshots, isolated_workspace, snapshot_tree)

from .fakes import BACKEND_DIR, FakeFAISS, import_paper_manager, make_fake_pipeline

PAPER_TEXT = "\n".join([
    "Automatic license plate recognition plays an important role",          # page 1
    "the overall rate of success for our LPR algorithm is 93.7%",            # page 2
    "only four distinct colors white black red and green are utilized",      # page 3
    "unrelated acknowledgement text",                                         # page 4
])

DATASET = {
    "name": "fake-ds",
    "status": "draft",
    "papers": [{"file": "paper.pdf"}],
    "items": [
        {"id": "rate", "question": "What is the overall rate of success of the LPR algorithm?",
         "expected_answer": "93.7%", "key_facts": ["93.7%"],
         "relevant_sources": [{"paper": "paper.pdf", "page": 2,
                               "evidence": "the overall rate of success for our LPR algorithm is 93.7%"}]},
        {"id": "greet", "question": "hello there",
         "expected_answer": "n/a", "relevant_sources": []},
    ],
}


class Base(unittest.TestCase):
    def setUp(self):
        self.pm = import_paper_manager()
        self._saved = (self.pm.BASE_DIR, self.pm.METADATA_DIR, self.pm.UPLOADS_BASE_DIR)
        # A fake "production" data folder with another user's session in it.
        self.real = tempfile.mkdtemp(prefix="fake-backend-")
        self.pm.BASE_DIR = self.real
        self.pm.METADATA_DIR = os.path.join(self.real, "metadata")
        self.pm.UPLOADS_BASE_DIR = os.path.join(self.real, "uploads")
        other = "user-session-12345678"
        os.makedirs(os.path.join(self.real, "uploads", other))
        os.makedirs(os.path.join(self.real, "vectorstore", other))
        with open(os.path.join(self.real, "uploads", other, "theirs.pdf"), "w") as f:
            f.write("someone else's paper")
        with open(os.path.join(self.real, "vectorstore", other, "index.faiss"), "w") as f:
            f.write("[]")
        os.makedirs(self.pm.METADATA_DIR)
        with open(os.path.join(self.pm.METADATA_DIR, f"{other}.json"), "w") as f:
            json.dump([{"paper_id": 1, "name": "theirs.pdf",
                        "path": os.path.join(self.real, "uploads", other, "theirs.pdf")}], f)
        self.other = other

        self.papers_dir = tempfile.mkdtemp(prefix="papers-")
        with open(os.path.join(self.papers_dir, "paper.pdf"), "w") as f:
            f.write(PAPER_TEXT)
        self.pipe = make_fake_pipeline(self.pm)

    def tearDown(self):
        self.pm.BASE_DIR, self.pm.METADATA_DIR, self.pm.UPLOADS_BASE_DIR = self._saved
        shutil.rmtree(self.real, ignore_errors=True)
        shutil.rmtree(self.papers_dir, ignore_errors=True)

    def run_eval(self, **cfg):
        return run_evaluation(DATASET, DATASET["items"], [os.path.join(self.papers_dir, "paper.pdf")],
                              self.pm, self.pipe, FakeFAISS,
                              RunConfig(k_values=(1, 3), progress=lambda *_: None, **cfg),
                              embed=self.pipe.embedding_model.embed_query)


class IsolationTests(Base):
    def test_existing_papers_and_indexes_untouched(self):
        before = snapshot_tree([self.real, self.papers_dir])
        repo_backend_before = snapshot_tree([BACKEND_DIR])
        result = self.run_eval()
        self.assertEqual(diff_snapshots(before, snapshot_tree([self.real, self.papers_dir])),
                         {"added": [], "removed": [], "modified": []})
        self.assertEqual(diff_snapshots(repo_backend_before, snapshot_tree([BACKEND_DIR])),
                         {"added": [], "removed": [], "modified": []})
        self.assertTrue(result["integrity"]["unchanged"])
        # Paths restored afterwards
        self.assertEqual(self.pm.BASE_DIR, self.real)
        self.assertEqual(self.pipe.BASE_DIR, self.real)

    def test_other_session_still_isolated_and_visible(self):
        self.run_eval()
        papers = self.pm.get_uploaded_papers(self.other)
        self.assertEqual([p["name"] for p in papers], ["theirs.pdf"])
        # and the evaluation's own session is gone with its workspace
        self.assertEqual(sorted(os.listdir(self.pm.METADATA_DIR)), [f"{self.other}.json"])

    def test_eval_session_cannot_see_real_sessions(self):
        with isolated_workspace(self.pm, self.pipe) as ws:
            self.assertNotEqual(self.pm.BASE_DIR, self.real)
            self.assertEqual(self.pm.get_uploaded_papers(self.other), [])
            self.assertTrue(os.path.realpath(self.pm.UPLOADS_BASE_DIR).startswith(os.path.realpath(ws)))
        self.assertFalse(os.path.exists(ws))

    def test_paths_restored_on_exception(self):
        with self.assertRaises(RuntimeError):
            with isolated_workspace(self.pm, self.pipe):
                raise RuntimeError("fail mid-run")
        self.assertEqual(self.pm.BASE_DIR, self.real)
        self.assertEqual(self.pipe.BASE_DIR, self.real)

    def test_keep_workspace(self):
        result = self.run_eval(keep_workspace=True)
        ws = result["run"]["kept_workspace"]
        try:
            self.assertTrue(os.path.isdir(ws))
            self.assertTrue(os.listdir(os.path.join(ws, "vectorstore")))
        finally:
            shutil.rmtree(ws, ignore_errors=True)


class InstrumentationTests(Base):
    def test_patches_installed_and_removed(self):
        original_client = self.pipe.client
        original_sim = FakeFAISS.similarity_search_with_score
        with PipelineInstrumentation(self.pipe, FakeFAISS):
            self.assertIsNot(self.pipe.client, original_client)
            self.assertIsNot(FakeFAISS.similarity_search_with_score, original_sim)
        self.assertIs(self.pipe.client, original_client)
        self.assertIs(FakeFAISS.similarity_search_with_score, original_sim)

    def test_nested_search_calls_recorded_once(self):
        vs = FakeFAISS([])
        with PipelineInstrumentation(self.pipe, FakeFAISS) as instr:
            with instr.capture() as trace:
                vs.similarity_search("q", k=2)  # calls similarity_search_with_score inside
        self.assertEqual([c.method for c in trace.retrieval_calls], ["similarity_search"])

    def test_extract_context_templates(self):
        self.assertEqual(extract_context("x\nResearch Context:\nCTX\n\nUser Question:\nq"), "CTX")
        self.assertEqual(extract_context("x\nContext:\nMETA\n\nQuestion:\nq\n\nAnswer:"), "META")
        self.assertEqual(extract_context("unknown template"), "unknown template")


class RunnerTests(Base):
    def test_metrics_and_routes(self):
        result = self.run_eval()
        rate, greet = result["results"]
        self.assertEqual(rate["pipeline_status"], "answered")
        self.assertEqual(rate["routes"], ["PAPER_CONTENT"])
        self.assertEqual(rate["recall@1"], 1.0)
        self.assertEqual(rate["context_recall"], 1.0)
        self.assertEqual(rate["key_fact_coverage"], 1.0)
        self.assertEqual(rate["unsupported_numbers"], [])
        self.assertEqual(rate["n_llm_calls"], 1)
        self.assertGreaterEqual(rate["latency_total_s"],
                                rate["latency_retrieval_s"] + rate["latency_generation_s"])
        self.assertTrue(any(c["relevant"] and c["in_llm_context"] for c in rate["retrieved_chunks"]))

        self.assertEqual(greet["pipeline_status"], "no_generation")
        self.assertEqual(greet["routes"], ["CONVERSATIONAL"])
        self.assertIsNone(greet["recall@1"])  # no relevant sources -> undefined
        self.assertIsNone(greet["latency_generation_s"])

        s = result["summary"]
        self.assertEqual((s["n_questions"], s["n_answered"], s["n_no_generation"]), (2, 1, 1))
        self.assertEqual(s["recall@1"]["n"], 1)
        self.assertTrue(all(e["ok"] for e in result["evidence_check"]))

    def test_hallucinated_number_flagged(self):
        self.pipe = make_fake_pipeline(self.pm, answer_fn=lambda kw: "Success was 99.9%.")
        rate = self.run_eval()["results"][0]
        self.assertEqual(rate["unsupported_numbers"], ["99.9%"])
        self.assertEqual(rate["key_fact_coverage"], 0.0)

    def test_pipeline_error_string_is_not_scored(self):
        self.pipe = make_fake_pipeline(self.pm, answer_fn=lambda kw: "x")
        self.pipe.ask_question = lambda sid, q: "Error during retrieval: something broke"
        result = self.run_eval(retries=0)
        self.assertEqual(result["summary"]["n_errors"], 2)
        self.assertEqual(result["summary"]["latency_total_s"]["n"], 0)
        self.assertEqual(result["summary"]["key_fact_coverage"]["n"], 0)
        self.assertEqual(result["summary"]["embedding_relevance"]["n"], 0)
        # retrieval still happened and is still measured
        self.assertEqual(result["summary"]["recall@1"]["n"], 1)

    def test_reports_written(self):
        result = self.run_eval()
        out = tempfile.mkdtemp()
        try:
            paths = write_all(result, out)
            with open(paths["json"]) as f:
                self.assertEqual(json.load(f)["summary"]["n_questions"], 2)
            with open(paths["csv"]) as f:
                lines = f.read().splitlines()
            self.assertIn("recall@1", lines[0])
            self.assertEqual(len(lines), 3)
            with open(paths["report"]) as f:
                md = f.read()
            for text in ("Questions evaluated: **2**", "Retrieval Recall@1", "not marked as human-verified",
                         "No file under the protected paths"):
                self.assertIn(text, md)
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
