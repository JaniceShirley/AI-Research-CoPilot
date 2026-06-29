from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

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
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"

@app.get("/")
def home():
    return {"message": "AI Research Backend Running 🚀"}


@app.post("/upload")

async def upload_pdf(file: UploadFile = File(...)):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    register_uploaded_paper(file.filename, file_path)

    total_chunks = process_pdf(file_path)

    return {
        "message": f"✅ {file.filename} processed successfully • {total_chunks} chunks created",
        "chunks": total_chunks
    }


@app.get("/ask")
def ask(question: str):

    answer = ask_question(question)

    return {
        "answer": answer
    }


@app.get("/compare")
def compare():

    result = compare_papers()

    return {
        "comparison": result
    }

@app.get("/ideas")
def ideas():

    result = generate_research_ideas()

    return {
        "ideas": result
    }


@app.get("/recommend")
def recommend():

    result = recommend_related_papers()

    return {
        "papers": result
    }

@app.delete("/paper/{paper_name}")
def delete_paper(paper_name: str):

    removed = remove_uploaded_paper(paper_name)

    if removed:
        rebuild_vectorstore()
        return {
            "message": f"{paper_name} removed successfully."
        }

    return {
        "message": "Paper not found."
    }