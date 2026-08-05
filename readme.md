# 🧠 AI Research Co-Pilot

AI Research Co-Pilot is a research assistant that lets you upload academic papers, index their content, and ask questions using a RAG (Retrieval-Augmented Generation) workflow.

This repository combines:
- a Python backend for PDF processing, vector indexing, and LLM orchestration
- a React + Vite frontend for a chat-style research assistant interface

## 🚀 Features
- Upload PDF research papers
- Extract and chunk paper text for semantic indexing
- Build FAISS vector search over uploaded papers
- Answer metadata questions using first-page paper context
- Route conversational, metadata, and paper-content questions correctly
- Support multi-paper uploads, comparison, and recommendations

## 🛠️ Tech Stack
- Python
- FastAPI
- Vite + React
- Tailwind CSS
- FAISS
- Sentence Transformers embeddings
- Groq / Llama for chat completions

## 📁 Repository layout
- `backend/` — FastAPI backend, PDF processing, vector indexing, retrieval logic
- `frontend/` — React/Vite UI
- `requirements.txt` — Python dependencies
- `frontend/package.json` — frontend dependencies
- `app.py` — legacy Streamlit prototype (not the current web app)

## ⚙️ Local setup
### 1. Install Python dependencies
```bash
python3 -m pip install -r requirements.txt
```

### 2. Install frontend dependencies
```bash
cd frontend
npm install
```

### 3. Create a `.env` file
Create a `.env` file in your home project folder or backend environment with at least:
```text
GROQ_API_KEY=your_groq_api_key
```
Optionally:
```text
HF_TOKEN=your_huggingface_token
```

### 4. Start the backend
```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Start the frontend
```bash
cd ../frontend
npm run dev
```

### 6. Open the app
- Frontend: `http://localhost:5173`
- Backend docs: `http://127.0.0.1:8000/docs`

## 💡 Notes
- Do not check `.env` into version control.
- The backend currently requires `GROQ_API_KEY` at runtime.
- `app.py` is an older Streamlit prototype and is not the main application surface today.
- PDF files are stored under `backend/uploads` when uploaded through the app.

## 🧪 Recommended workflow
1. Start backend and frontend.
2. Upload one or more PDFs.
3. Ask metadata or paper-content questions.
4. Use the debug logs in the backend terminal to verify query routing when needed.

## 📝 License
MIT License