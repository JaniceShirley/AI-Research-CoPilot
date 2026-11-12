"""Offline evaluation framework for the AI Research Co-Pilot RAG pipeline.

This package lives outside ``backend/`` on purpose: it calls the production
``rag_pipeline.ask_question`` unchanged and only observes it (timing and
capturing FAISS / Groq calls from the outside). Nothing in here is imported by
the FastAPI app.

Run ``python -m evaluation --help`` from the repository root.
"""

__all__ = []
