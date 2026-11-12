"""Run the pipeline against a throw-away copy of the backend's data folders.

The backend stores everything under ``backend/{uploads,metadata,vectorstore}``
keyed by session ID. During an evaluation we point those module-level paths at
a temporary directory, so the evaluation:

* never reads, writes, re-indexes or deletes a real user's papers or FAISS
  index, and
* gets its own session ID, so the existing per-session isolation still holds
  inside the temporary workspace.

``snapshot_tree`` / ``diff_snapshots`` are used to prove (in tests and in every
report) that the real data folders were left untouched.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import secrets
import shutil
import tempfile
from typing import Any, Iterable, Optional


def new_eval_session_id() -> str:
    # Matches paper_manager.validate_session_id: ^[a-zA-Z0-9_\-]{8,128}$
    return f"eval-{secrets.token_hex(8)}"


@contextlib.contextmanager
def isolated_workspace(paper_manager: Any, pipeline: Any, keep: bool = False,
                       root: Optional[str] = None):
    """Temporarily redirect the backend's storage paths into a temp dir.

    Yields the temp directory path. Original paths are always restored.
    """
    tmp = tempfile.mkdtemp(prefix="rag-eval-", dir=root)
    saved_pm = {
        name: getattr(paper_manager, name)
        for name in ("BASE_DIR", "METADATA_DIR", "UPLOADS_BASE_DIR")
    }
    saved_pipe = getattr(pipeline, "BASE_DIR", None)
    try:
        paper_manager.BASE_DIR = tmp
        paper_manager.METADATA_DIR = os.path.join(tmp, "metadata")
        paper_manager.UPLOADS_BASE_DIR = os.path.join(tmp, "uploads")
        if saved_pipe is not None:
            pipeline.BASE_DIR = tmp  # used by get_session_vectorstore_paths
        yield tmp
    finally:
        for name, value in saved_pm.items():
            setattr(paper_manager, name, value)
        if saved_pipe is not None:
            pipeline.BASE_DIR = saved_pipe
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)


def ingest_papers(paper_manager: Any, pipeline: Any, session_id: str,
                  pdf_paths: Iterable[str]) -> dict:
    """Upload papers through the production code path, in the given order.

    Mirrors ``POST /upload``: copy the PDF into the session's upload folder,
    ``register_uploaded_paper`` and ``process_pdf`` (which rebuilds the
    session's FAISS index). The source PDFs are only read.
    Returns {"papers": [...], "chunks": int}.
    """
    upload_dir, _ = paper_manager.get_session_paths(session_id)
    os.makedirs(upload_dir, exist_ok=True)
    total = 0
    for src in pdf_paths:
        dest = os.path.join(upload_dir, os.path.basename(src))
        shutil.copyfile(src, dest)
        paper_manager.register_uploaded_paper(session_id, os.path.basename(src), dest)
        total = pipeline.process_pdf(session_id, dest)
    return {"papers": paper_manager.get_uploaded_papers(session_id), "chunks": total}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_tree(paths: Iterable[str]) -> dict:
    """Map every file under ``paths`` to (size, mtime_ns, sha256)."""
    snap = {}
    for root in paths:
        if not root or not os.path.exists(root):
            continue
        if os.path.isfile(root):
            st = os.stat(root)
            snap[os.path.abspath(root)] = (st.st_size, st.st_mtime_ns, file_sha256(root))
            continue
        for dirpath, _, files in os.walk(root):
            for fn in files:
                p = os.path.abspath(os.path.join(dirpath, fn))
                try:
                    st = os.stat(p)
                    snap[p] = (st.st_size, st.st_mtime_ns, file_sha256(p))
                except OSError:
                    continue
    return snap


def diff_snapshots(before: dict, after: dict) -> dict:
    return {
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "modified": sorted(p for p in set(before) & set(after) if before[p] != after[p]),
    }
