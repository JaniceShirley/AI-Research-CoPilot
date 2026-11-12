"""Optional live smoke test of the existing app features.

Skipped unless RAG_EVAL_LIVE=1 and RAG_EVAL_PAPER points at a PDF. It drives
the real FastAPI app (upload, papers, ask, compare, ideas, recommend, delete)
inside an isolated workspace, so no real session data is touched. It makes
real Groq calls and one Semantic Scholar search.

    RAG_EVAL_LIVE=1 RAG_EVAL_PAPER=/path/to/paper.pdf \
      python -m unittest evaluation.tests.test_live_app_smoke -v
"""

import os
import sys
import unittest

LIVE = os.getenv("RAG_EVAL_LIVE") == "1"
PAPER = os.getenv("RAG_EVAL_PAPER", "")


@unittest.skipUnless(LIVE and os.path.isfile(PAPER), "set RAG_EVAL_LIVE=1 and RAG_EVAL_PAPER=<pdf>")
class LiveAppSmokeTest(unittest.TestCase):
    def test_all_endpoints(self):
        from evaluation.cli import BACKEND_DIR, _load_env
        from evaluation.workspace import (diff_snapshots, isolated_workspace, new_eval_session_id,
                                          snapshot_tree)
        _load_env(os.getenv("RAG_EVAL_ENV_FILE"))
        if BACKEND_DIR not in sys.path:
            sys.path.insert(0, BACKEND_DIR)
        import main
        import paper_manager
        import rag_pipeline
        from fastapi.testclient import TestClient

        protected = [os.path.join(paper_manager.BASE_DIR, d) for d in ("uploads", "metadata", "vectorstore")]
        before = snapshot_tree(protected + [os.path.dirname(PAPER)])
        sid = new_eval_session_id()
        h = {"X-Session-ID": sid}
        name = os.path.basename(PAPER)
        with isolated_workspace(paper_manager, rag_pipeline):
            c = TestClient(main.app)
            self.assertEqual(c.get("/papers").status_code, 400)  # session header still required
            with open(PAPER, "rb") as f:
                r = c.post("/upload", files={"file": (name, f, "application/pdf")}, headers=h)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertGreater(r.json()["chunks"], 0)
            self.assertEqual([p["name"] for p in c.get("/papers", headers=h).json()["papers"]], [name])
            # A different session sees nothing
            self.assertEqual(c.get("/papers", headers={"X-Session-ID": new_eval_session_id()}).json()["papers"], [])

            for path, key in (("/ask?question=What+methodology+is+used%3F", "answer"),
                              ("/compare", "comparison"), ("/ideas", "ideas")):
                with self.subTest(endpoint=path):  # report every endpoint, not just the first failure
                    r = c.get(path, headers=h)
                    self.assertEqual(r.status_code, 200, r.text)
                    text = r.json()[key]
                    self.assertTrue(text and "Error" not in text[:40], f"{path}: {text[:300]}")
            with self.subTest(endpoint="/recommend"):
                r = c.get("/recommend", headers=h)
                self.assertEqual(r.status_code, 200)
                papers = r.json()["papers"]
                self.assertIsInstance(papers, list)
                self.assertFalse(any(p.get("error") for p in papers), str(papers)[:300])

            r = c.delete(f"/paper/{name}", headers=h)
            self.assertEqual(r.json()["papers"], [])
        self.assertEqual(diff_snapshots(before, snapshot_tree(protected + [os.path.dirname(PAPER)])),
                         {"added": [], "removed": [], "modified": []})


if __name__ == "__main__":
    unittest.main()
