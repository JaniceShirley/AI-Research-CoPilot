from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import re

from rag_pipeline import (
    process_pdf,
    ask_question,
    compare_papers,
    generate_research_ideas,
    recommend_related_papers,
    rebuild_vectorstore
)

from paper_manager import (
    register_uploaded_paper,
    remove_uploaded_paper,
    get_uploaded_papers,
    get_session_paths,
    validate_session_id
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_session_id(x_session_id: str = Header(None, alias="X-Session-ID")) -> str:
    """FastAPI Dependency to validate X-Session-ID header on incoming requests."""
    if not x_session_id or not x_session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Session-ID header is required."
        )
    try:
        clean_id = validate_session_id(x_session_id)
        return clean_id
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err)
        )


@app.get("/")
def home():
    return {"message": "AI Research Backend Running"}


@app.get("/papers")
def list_papers(session_id: str = Depends(get_session_id)):
    return {
        "papers": get_uploaded_papers(session_id)
    }


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id)
):
    upload_dir, _ = get_session_paths(session_id)
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    paper_entry = register_uploaded_paper(session_id, file.filename, file_path)

    total_chunks = process_pdf(session_id, file_path)

    return {
        "message": f"{file.filename} processed successfully • {total_chunks} chunks created",
        "chunks": total_chunks,
        "paper": paper_entry,
        "papers": get_uploaded_papers(session_id)
    }


@app.get("/ask")
def ask(
    question: str,
    session_id: str = Depends(get_session_id)
):
    answer = ask_question(session_id, question)
    return {
        "answer": answer
    }


@app.get("/compare")
def compare(session_id: str = Depends(get_session_id)):
    result = compare_papers(session_id)
    return {
        "comparison": result
    }


@app.get("/ideas")
def ideas(session_id: str = Depends(get_session_id)):
    result = generate_research_ideas(session_id)
    return {
        "ideas": result
    }


@app.get("/recommend")
def recommend(session_id: str = Depends(get_session_id)):
    result = recommend_related_papers(session_id)
    return {
        "papers": result
    }


@app.delete("/paper/{paper_name}")
def delete_paper(
    paper_name: str,
    session_id: str = Depends(get_session_id)
):
    removed = remove_uploaded_paper(session_id, paper_name)

    if removed:
        rebuild_vectorstore(session_id)
        return {
            "message": f"{paper_name} removed successfully.",
            "papers": get_uploaded_papers(session_id)
        }

    return {
        "message": "Paper not found.",
        "papers": get_uploaded_papers(session_id)
    }