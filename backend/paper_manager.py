import json
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS_FILE = os.path.join(BASE_DIR, "uploaded_papers.json")

def _create_file_if_missing():
    if not os.path.exists(PAPERS_FILE):
        with open(PAPERS_FILE, "w") as f:
            json.dump([], f, indent=4)


def load_uploaded_papers():
    _create_file_if_missing()

    with open(PAPERS_FILE, "r") as f:
        return json.load(f)


def save_uploaded_papers(papers):
    with open(PAPERS_FILE, "w") as f:
        json.dump(papers, f, indent=4)


def get_uploaded_papers():
    return load_uploaded_papers()


def register_uploaded_paper(filename, filepath):

    papers = load_uploaded_papers()

    # Don't register duplicate file paths
    for paper in papers:
        if paper["path"] == filepath:
            return

    next_id = 1

    if papers:
        next_id = max(p["paper_id"] for p in papers) + 1

    papers.append(
        {
            "paper_id": next_id,
            "name": filename,
            "path": filepath
        }
    )

    save_uploaded_papers(papers)


def remove_uploaded_paper(filename):
    papers = load_uploaded_papers()
    remaining = []
    removed = False
    for paper in papers:
        if paper["name"] == filename:
            removed = True
            # Delete PDF from uploads folder
            if os.path.exists(paper["path"]):
                os.remove(paper["path"])
        else:
            remaining.append(paper)
    save_uploaded_papers(remaining)
    return removed


def clear_uploaded_papers():
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
    return [paper["path"] for paper in papers]