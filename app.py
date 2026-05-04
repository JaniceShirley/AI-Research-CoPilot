import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
import os

st.set_page_config(page_title="AI Research Co-Pilot", layout="wide")

model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("AI Research Co-Pilot")

# Model selection
mode = st.radio("Select Mode", ["Offline (Ollama)", "Online (Groq)"])

# STEP 1: Upload PDF
uploaded_files = st.file_uploader("📄 Upload Research Papers", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} paper(s) uploaded successfully")

    chunk_size = 500
    chunks = []
    chunk_sources = []

    for file_idx, file in enumerate(uploaded_files):
        reader = PdfReader(file)
        file_text = ""

        for page in reader.pages:
            file_text += page.extract_text() or ""

        file_chunks = [file_text[i:i+chunk_size] for i in range(0, len(file_text), chunk_size)]

        for chunk in file_chunks:
            chunks.append(chunk)
            chunk_sources.append(f"Paper {file_idx+1}: {file.name}")

    if len(chunks) == 0:
        st.error("No readable text found in PDF.")
        st.stop()

    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))

    st.markdown("---")

    # STEP 2: Q&A Section
    st.subheader("💬 Ask Questions")

    st.markdown("---")
    st.subheader("🔍 Compare Papers & Detect Contradictions")
    compare_button = st.button("Compare Papers")

    question = st.text_input("Type your question here")

    if question:
        question = question.lower().strip()
        st.markdown(f"**👤 You:** {question}")

        with st.spinner("🤖 Thinking..."):
            q_embedding = model.encode([question]).astype('float32')
            D, I = index.search(np.array(q_embedding), k=8)

            relevant_chunks = [(chunks[i], chunk_sources[i]) for i in I[0]]
            context = "\n\n".join([chunk for chunk, _ in relevant_chunks[:5]])

            prompt = f"""
You are an intelligent AI research assistant.

Your job is to answer questions based on the given research paper context.

Instructions:
- Understand the context deeply before answering
- If multiple pieces of information exist, combine them logically
- Explain in simple and clear terms
- If the answer is not clearly present, say: "Not clearly mentioned in the paper"
- Do NOT copy text blindly, explain in your own words

Context:
{context}

Question:
{question}

Answer:
"""

            if mode == "Offline (Ollama)":
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3",
                        "prompt": prompt,
                        "stream": False
                    }
                )
                answer = response.json()["response"]

            else:
                groq_api_key = os.getenv("GROQ_API_KEY")

                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama3-8b-8192",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    }
                )

                answer = response.json()["choices"][0]["message"]["content"]

        # STEP 3: Show Answer
        st.markdown("### 🤖 Answer")
        st.success(answer)

        # STEP 4: Show Source
        st.markdown("### 📚 Source (Retrieved Context)")
        for i, (chunk, source) in enumerate(relevant_chunks):
            st.markdown(f"**Chunk {i+1} ({source}):**")
            st.write(chunk)
            st.markdown("---")

    # STEP 5: Paper Comparison & Contradiction
    if compare_button:
        with st.spinner("🔎 Analyzing papers..."):
            # Use top chunks from ALL documents (not dependent on question)
            comparison_context = "\n\n".join(chunks[:10])

            comparison_prompt = f"""
You are an AI research analyst.

Your task is to analyze multiple research papers and:
1. Identify key differences in approaches
2. Highlight any contradictions between papers
3. Summarize similarities

Be clear and structured.

Context from multiple papers:
{comparison_context}

Output format:
- Key Differences:
- Contradictions:
- Similarities:
"""

            if mode == "Offline (Ollama)":
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3",
                        "prompt": comparison_prompt,
                        "stream": False
                    }
                )
                comparison_answer = response.json()["response"]
            else:
                groq_api_key = os.getenv("GROQ_API_KEY")

                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama3-8b-8192",
                        "messages": [
                            {"role": "user", "content": comparison_prompt}
                        ]
                    }
                )

                comparison_answer = response.json()["choices"][0]["message"]["content"]

        st.markdown("### 📊 Paper Comparison & Contradictions")
        st.info(comparison_answer)