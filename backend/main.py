from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from rag_pipeline import process_pdf, ask_question, compare_papers, generate_research_ideas

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