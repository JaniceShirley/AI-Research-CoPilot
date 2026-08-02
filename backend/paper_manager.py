import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS_FILE = os.path.join(BASE_DIR, "uploaded_papers.json")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

def _create_file_if_missing():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    if not os.path.exists(PAPERS_FILE):
        with open(PAPERS_FILE, "w") as f:
            json.dump([], f, indent=4)


def resolve_paper_path(path_str):
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(BASE_DIR, path_str)


def load_uploaded_papers():
    _create_file_if_missing()
    try:
        with open(PAPERS_FILE, "r") as f:
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
            save_uploaded_papers(valid_papers)

        return valid_papers
    except Exception:
        return []


def save_uploaded_papers(papers):
    _create_file_if_missing()
    with open(PAPERS_FILE, "w") as f:
        json.dump(papers, f, indent=4)


def get_uploaded_papers():
    return load_uploaded_papers()


def register_uploaded_paper(filename, filepath):
    papers = load_uploaded_papers()

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
    save_uploaded_papers(papers)
    return paper_entry


def remove_uploaded_paper(filename):
    papers = load_uploaded_papers()
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

    save_uploaded_papers(remaining)
    return removed


def clear_uploaded_papers():
    papers = load_uploaded_papers()
    for paper in papers:
        abs_path = resolve_paper_path(paper["path"])
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
            except OSError:
                pass
    save_uploaded_papers([])


def paper_exists(filename):
    papers = load_uploaded_papers()
    for paper in papers:
        if paper["name"] == filename:
            return True
    return False


def get_paper_count():
    return len(load_uploaded_papers())


def get_paper_names():
    papers = load_uploaded_papers()
    return [paper["name"] for paper in papers]


def get_paper_paths():
    papers = load_uploaded_papers()
    return [resolve_paper_path(paper["path"]) for paper in papers]