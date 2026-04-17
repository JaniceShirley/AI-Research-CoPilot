import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

st.title("🧠 AI Research Co-Pilot")

# Upload PDF
uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully!")

    # Read PDF
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    # --- Create embeddings ---
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Split text into chunks
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    st.write(f"Total chunks created: {len(chunks)}")

    # Convert chunks to embeddings
    embeddings = model.encode(chunks)

    st.success("Embeddings created successfully!")

    # --- Create FAISS index ---
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Show extracted text (preview)
    st.subheader("📄 Extracted Text Preview")
    st.write(text[:1000])  # show only first 1000 characters

    # Ask question
    question = st.text_input("Ask a question about the paper:")

    if question:
        st.write("You asked:", question)

        # Convert question to embedding
        q_embedding = model.encode([question])

        # Search similar chunks
        D, I = index.search(np.array(q_embedding), k=3)

        # Get relevant chunks
        relevant_chunks = [chunks[i] for i in I[0]]

        st.subheader("🔍 Relevant Context")
        for chunk in relevant_chunks:
            st.write(chunk)

        # Simple answer (combine chunks)
        answer = " ".join(relevant_chunks)

        st.subheader("🤖 Answer")
        st.write(answer)