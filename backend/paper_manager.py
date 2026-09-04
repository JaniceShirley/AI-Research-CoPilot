import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
UPLOADS_BASE_DIR = os.path.join(BASE_DIR, "uploads")


def validate_session_id(session_id: str) -> str:
    """Validate and sanitize session_id to prevent path traversal."""
    if not session_id or not isinstance(session_id, str):
        raise ValueError("Session ID is required.")
    clean_id = session_id.strip()
    if not re.match(r"^[a-zA-Z0-9_\-]{8,128}$", clean_id):
        raise ValueError("Invalid Session ID format.")
    return clean_id


def get_session_paths(session_id: str):
    clean_id = validate_session_id(session_id)
    upload_dir = os.path.join(UPLOADS_BASE_DIR, clean_id)
    metadata_file = os.path.join(METADATA_DIR, f"{clean_id}.json")
    return upload_dir, metadata_file


def _create_session_dirs_if_missing(session_id: str):
    upload_dir, metadata_file = get_session_paths(session_id)
    os.makedirs(METADATA_DIR, exist_ok=True)
    os.makedirs(upload_dir, exist_ok=True)
    if not os.path.exists(metadata_file):
        with open(metadata_file, "w") as f:
            json.dump([], f, indent=4)
    return upload_dir, metadata_file


def resolve_paper_path(path_str):
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(BASE_DIR, path_str)


def load_uploaded_papers(session_id: str):
    try:
        upload_dir, metadata_file = _create_session_dirs_if_missing(session_id)
        with open(metadata_file, "r") as f:
            papers = json.load(f)

        valid_papers = []
        modified = False
        for paper in papers:
            abs_path = resolve_paper_path(paper["path"])
            if os.path.exists(abs_path):
                valid_papers.append(paper)
            else:
                modified = True

        if modified:
            for index, p in enumerate(valid_papers, start=1):
                p["paper_id"] = index
            save_uploaded_papers(session_id, valid_papers)

        return valid_papers
    except Exception:
        return []


def save_uploaded_papers(session_id: str, papers: list):
    _, metadata_file = _create_session_dirs_if_missing(session_id)
    with open(metadata_file, "w") as f:
        json.dump(papers, f, indent=4)


def get_uploaded_papers(session_id: str):
    return load_uploaded_papers(session_id)


def register_uploaded_paper(session_id: str, filename: str, filepath: str):
    papers = load_uploaded_papers(session_id)

    # If paper already exists with same filename or path, return existing record
    for paper in papers:
        if paper["name"] == filename or paper["path"] == filepath:
            return paper

    next_id = len(papers) + 1

    paper_entry = {
        "paper_id": next_id,
        "name": filename,
        "path": filepath
    }

    papers.append(paper_entry)
    save_uploaded_papers(session_id, papers)
    return paper_entry


def remove_uploaded_paper(session_id: str, filename: str):
    papers = load_uploaded_papers(session_id)
    remaining = []
    removed = False

    for paper in papers:
        if paper["name"] == filename:
            removed = True
            abs_path = resolve_paper_path(paper["path"])
            if os.path.exists(abs_path):
                try:
                    os.remove(abs_path)
                except OSError:
                    pass
        else:
            remaining.append(paper)

    # Re-index remaining paper_ids sequentially 1..N
    for index, paper in enumerate(remaining, start=1):
        paper["paper_id"] = index

    save_uploaded_papers(session_id, remaining)
    return removed


def clear_uploaded_papers(session_id: str):
    papers = load_uploaded_papers(session_id)
    for paper in papers:
        abs_path = resolve_paper_path(paper["path"])
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
            except OSError:
                pass
    save_uploaded_papers(session_id, [])


def paper_exists(session_id: str, filename: str):
    papers = load_uploaded_papers(session_id)
    for paper in papers:
        if paper["name"] == filename:
            return True
    return False


def get_paper_count(session_id: str):
    return len(load_uploaded_papers(session_id))


def get_paper_names(session_id: str):
    papers = load_uploaded_papers(session_id)
    return [paper["name"] for paper in papers]


def get_paper_paths(session_id: str):
    papers = load_uploaded_papers(session_id)
    return [resolve_paper_path(paper["path"]) for paper in papers]