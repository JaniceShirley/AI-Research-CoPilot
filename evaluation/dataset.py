"""Load and validate the evaluation dataset (JSON)."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

REQUIRED_ITEM_FIELDS = ("id", "question", "expected_answer", "relevant_sources")


class DatasetError(ValueError):
    pass


def load_dataset(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise DatasetError(f"{path}: invalid JSON: {e}") from e
    validate_dataset(data)
    data["_path"] = os.path.abspath(path)
    data["_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return data


def validate_dataset(data: dict) -> None:
    """Raise DatasetError describing every schema problem found."""
    problems = []
    if not isinstance(data, dict):
        raise DatasetError("dataset must be a JSON object")
    papers = data.get("papers")
    if not isinstance(papers, list) or not papers:
        problems.append("'papers' must be a non-empty list")
        papers = []
    names = []
    for i, p in enumerate(papers):
        if not isinstance(p, dict) or not p.get("file"):
            problems.append(f"papers[{i}] needs a 'file' (PDF file name)")
        else:
            names.append(os.path.basename(p["file"]).lower())
    if len(set(names)) != len(names):
        problems.append("papers contains duplicate file names")

    items = data.get("items")
    if not isinstance(items, list) or not items:
        problems.append("'items' must be a non-empty list")
        items = []
    ids = set()
    for i, it in enumerate(items):
        where = f"items[{i}]"
        if not isinstance(it, dict):
            problems.append(f"{where} must be an object")
            continue
        for fld in REQUIRED_ITEM_FIELDS:
            if fld not in it:
                problems.append(f"{where} missing '{fld}'")
        if it.get("id") in ids:
            problems.append(f"{where} duplicate id {it.get('id')!r}")
        ids.add(it.get("id"))
        if not str(it.get("question", "")).strip():
            problems.append(f"{where} has an empty question")
        srcs = it.get("relevant_sources", [])
        if not isinstance(srcs, list):
            problems.append(f"{where}.relevant_sources must be a list")
            continue
        for j, s in enumerate(srcs):
            sw = f"{where}.relevant_sources[{j}]"
            if not isinstance(s, dict):
                problems.append(f"{sw} must be an object")
                continue
            if not s.get("paper"):
                problems.append(f"{sw} missing 'paper'")
            elif names and os.path.basename(s["paper"]).lower() not in names:
                problems.append(f"{sw} paper {s['paper']!r} is not listed in 'papers'")
            if not str(s.get("evidence", "")).strip():
                problems.append(f"{sw} missing 'evidence' (a verbatim quote from the paper)")
            if s.get("page") is not None and not (isinstance(s["page"], int) and s["page"] >= 1):
                problems.append(f"{sw} 'page' must be a 1-based integer or omitted")
        kf = it.get("key_facts", [])
        if kf is not None and not (isinstance(kf, list) and all(isinstance(x, str) for x in kf)):
            problems.append(f"{where}.key_facts must be a list of strings")
    if problems:
        raise DatasetError("dataset is invalid:\n  - " + "\n  - ".join(problems))


def resolve_paper_files(data: dict, papers_dir: str) -> list:
    """Return absolute PDF paths in dataset order; raise if any is missing."""
    out, missing = [], []
    for p in data["papers"]:
        path = os.path.join(papers_dir, os.path.basename(p["file"]))
        if os.path.isfile(path):
            out.append(os.path.abspath(path))
        else:
            missing.append(path)
    if missing:
        raise DatasetError("missing PDF file(s):\n  - " + "\n  - ".join(missing))
    return out


def select_items(data: dict, ids: Optional[list] = None, limit: Optional[int] = None) -> list:
    items = data["items"]
    if ids:
        wanted = set(ids)
        items = [it for it in items if it["id"] in wanted]
        unknown = wanted - {it["id"] for it in items}
        if unknown:
            raise DatasetError(f"unknown item id(s): {sorted(unknown)}")
    if limit is not None:
        items = items[:limit]
    return items
