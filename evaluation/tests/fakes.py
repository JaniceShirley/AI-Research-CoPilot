"""A stand-in for ``backend/rag_pipeline.py`` that needs no Groq key, model
download or FAISS. It writes to disk through the same BASE_DIR-based paths as
the real module, so workspace isolation is exercised for real."""

import json
import os
import sys
import types
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")


def import_paper_manager():
    """The real backend/paper_manager.py (standard library only)."""
    if BACKEND_DIR not in sys.path:
        sys.path.insert(0, BACKEND_DIR)
    import paper_manager
    return paper_manager


class Doc:
    def __init__(self, text, source, page, paper_id=1):
        self.page_content = text
        self.metadata = {"source": source, "page": page, "paper_id": paper_id}


class FakeFAISS:
    """Mimics the parts of langchain FAISS the pipeline uses."""

    def __init__(self, docs):
        self.docs = docs
        self.docstore = SimpleNamespace(_dict={str(i): d for i, d in enumerate(docs)})

    @classmethod
    def load_local(cls, folder, embeddings, allow_dangerous_deserialization=False):
        with open(os.path.join(folder, "index.faiss")) as f:
            raw = json.load(f)
        return cls([Doc(**d) for d in raw])

    def similarity_search_with_score(self, query, k=4, **kwargs):
        q = set(query.lower().split())
        scored = sorted(self.docs, key=lambda d: -len(q & set(d.page_content.lower().split())))
        return [(d, float(i)) for i, d in enumerate(scored[:k])]

    def similarity_search(self, query, k=4, **kwargs):
        # Like the real FAISS: delegates to similarity_search_with_score.
        return [d for d, _ in self.similarity_search_with_score(query, k, **kwargs)]

    def max_marginal_relevance_search(self, query, k=4, fetch_k=20, **kwargs):
        return self.similarity_search(query, k)


def make_fake_pipeline(paper_manager, answer_fn=None, raise_on=None):
    """Build a module-like object with the same surface as rag_pipeline."""
    mod = types.SimpleNamespace()
    mod.BASE_DIR = paper_manager.BASE_DIR
    mod.FAISS = FakeFAISS
    mod.GROQ_MODEL = "fake-model"
    mod.embedding_model = SimpleNamespace(model_name="fake-embed",
                                          embed_query=lambda t: [len(t), 1.0])
    mod.llm_calls = []

    def create(**kwargs):
        mod.llm_calls.append(kwargs)
        text = answer_fn(kwargs) if answer_fn else "The paper reports 93.7% overall success."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    mod.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def get_session_vectorstore_paths(session_id):
        v_dir = os.path.join(mod.BASE_DIR, "vectorstore", session_id)
        return v_dir, os.path.join(v_dir, "index.faiss")

    def process_pdf(session_id, pdf_path):
        v_dir, idx = get_session_vectorstore_paths(session_id)
        os.makedirs(v_dir, exist_ok=True)
        docs = []
        for p in paper_manager.get_uploaded_papers(session_id):
            with open(p["path"]) as f:
                for page, line in enumerate(f.read().splitlines()):
                    docs.append({"text": line, "source": p["name"], "page": page,
                                 "paper_id": p["paper_id"]})
        with open(idx, "w") as f:
            json.dump(docs, f)
        return len(docs)

    def ask_question(session_id, question):
        if raise_on and raise_on in question:
            raise RuntimeError("boom")
        v_dir, idx = get_session_vectorstore_paths(session_id)
        if not os.path.exists(idx):
            return "No research papers uploaded yet."
        if "hello" in question.lower():
            print("Query route: CONVERSATIONAL")
            return "Hello — I'm an AI Research CoPilot."
        print("Query route: PAPER_CONTENT")
        vs = mod.FAISS.load_local(v_dir, mod.embedding_model, allow_dangerous_deserialization=True)
        docs = vs.max_marginal_relevance_search(question, k=4, fetch_k=8)
        context = "\n\n".join(f"Source: {d.metadata['source']}\nContent: {d.page_content}"
                              for d in docs[:2])
        prompt = f"Instructions\n\nResearch Context:\n{context}\n\nUser Question:\n{question}\n"
        resp = mod.client.chat.completions.create(model=mod.GROQ_MODEL,
                                                  messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content

    mod.get_session_vectorstore_paths = get_session_vectorstore_paths
    mod.process_pdf = process_pdf
    mod.ask_question = ask_question
    return mod
