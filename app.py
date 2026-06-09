import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="AI Research Co-Pilot", layout="wide")

# Custom CSS styling
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}
.stTextInput>div>div>input {
    background-color: #1e222b;
    color: white;
}
.stButton>button {
    border-radius: 10px;
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("""
<h1 style='text-align: center;'>🧠 AI Research Co-Pilot</h1>
<p style='text-align: center; color: gray;'>Analyze, compare and understand research papers intelligently</p>
""", unsafe_allow_html=True)

# Model selection
mode = st.radio("Select Mode", ["Offline (Ollama)", "Online (Groq)"])

# STEP 1: Upload PDF
uploaded_files = st.file_uploader("📄 Upload Research Papers", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} paper(s) uploaded successfully")

    chunk_size = 800
    chunks = []
    chunk_sources = []
    paper_chunks = {}

    for file_idx, file in enumerate(uploaded_files):
        reader = PdfReader(file)
        file_text = ""

        for page in reader.pages:
            file_text += page.extract_text() or ""

        file_chunks = [file_text[i:i+chunk_size] for i in range(0, len(file_text), chunk_size)]
        paper_chunks[file.name] = file_chunks

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

    col1, col2 = st.columns(2)

    # LEFT COLUMN → Q&A
    with col1:
        st.subheader("💬 Ask Questions")

        question = st.text_input(
            "Ask about the papers",
            placeholder="Ask about methods, models, limitations, results..."
        )

        ask_button = st.button("🚀 Ask AI")

        # Display chat history
        for chat in st.session_state.chat_history:
            st.markdown(f"""
            <div style='background-color:#1e222b;padding:10px;border-radius:10px;margin-bottom:10px;'>
            👤 <b>You:</b> {chat['question']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background-color:#262730;padding:15px;border-radius:10px;margin-bottom:10px;'>
            🤖 <b>AI:</b><br>{chat['answer']}
            </div>
            """, unsafe_allow_html=True)

    # RIGHT COLUMN → Comparison
    with col2:
        st.subheader("🔍 Compare Papers")

        compare_button = st.button("📊 Compare Papers")
        suggest_button = st.button("💡 Generate Research Ideas")

    if ask_button and question:
        question = question.lower().strip()

        with st.spinner("🤖 Thinking..."):
            q_embedding = model.encode([question]).astype('float32')
            D, I = index.search(np.array(q_embedding), k=5)

            relevant_chunks = []

            for i in I[0]:
                chunk = chunks[i]
                source = chunk_sources[i]

                # Boost relevance if paper keywords appear in question
                question_lower = question.lower()

                if (
                    ("yolo" in question_lower and "yolo" in chunk.lower()) or
                    ("tgcn" in question_lower and "tgcn" in chunk.lower()) or
                    ("graph" in question_lower and "graph" in chunk.lower()) or
                    ("parking" in question_lower and "parking" in chunk.lower())
                ):
                    relevant_chunks.insert(0, (chunk, source))
                else:
                    relevant_chunks.append((chunk, source))

            context = ""
            used_sources = set()

            for chunk, source in relevant_chunks[:5]:
                if source not in used_sources:
                    context += f"\n\n===== {source} =====\n"
                    used_sources.add(source)

                context += chunk + "\n"

            comparison_keywords = ["compare", "difference", "vs", "versus", "similarity", "contradiction"]
            is_comparison_question = any(word in question.lower() for word in comparison_keywords)

            # If comparison-type question, build paper-wise context
            if is_comparison_question:
                comparison_context = ""

                for paper_name, paper_chunk_list in paper_chunks.items():
                    comparison_context += f"\n\n===== PAPER: {paper_name} =====\n"
                    comparison_context += "\n".join(paper_chunk_list[:4])

                context = comparison_context

            prompt = f"""
You are an intelligent multi-paper AI research assistant.

You MUST answer ONLY using the provided research paper context.
If the question is a comparison or contradiction question, analyze ALL uploaded papers together and compare them technically.

Rules:
- Do not invent information.
- If the paper does not explicitly mention something, clearly say:
  'This is not clearly described in the paper.'
- Focus carefully on technical workflow, methodology, architecture, and pipeline.
- If multiple papers are uploaded, identify which paper the question refers to.
- Use only the most relevant paper chunks while answering.
- Clearly mention the paper/source when answering.
- If the question compares papers, analyze all uploaded papers together.
- Mention technical differences, similarities, and contradictions explicitly.
- Never answer using only one paper for comparison questions.
- Give concise but technically correct answers.
- Summarize workflows step-by-step if available.

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
                print("API KEY FOUND:", groq_api_key is not None)

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

                response_json = response.json()
                print("Groq Response:", response_json)

                if "choices" in response_json:
                    answer = response_json["choices"][0]["message"]["content"]
                else:
                    answer = f"Groq API Error: {response_json}"

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

        with col1:
            st.markdown(f"""
            <div style='background-color:#1e222b;padding:10px;border-radius:10px;margin-bottom:10px;'>
            👤 <b>You:</b> {question}
            </div>
            """ , unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background-color:#262730;padding:15px;border-radius:10px;margin-bottom:10px;'>
            🤖 <b>AI:</b><br>{answer}
            </div>
            """, unsafe_allow_html=True)

    if compare_button:
        with st.spinner("🔎 Analyzing papers..."):
            # Build balanced paper-wise comparison context
            comparison_context = ""

            for paper_name, paper_chunk_list in paper_chunks.items():
                comparison_context += f"\n\n===== PAPER: {paper_name} =====\n"
                comparison_context += "\n".join(paper_chunk_list[:5])

            comparison_prompt = f"""
You are an advanced AI research analyst.

You are given multiple research papers.
Each paper is separated using headings like:
===== PAPER: paper_name =====

Your task:
1. Compare the methodologies used in each paper
2. Identify technical differences clearly
3. Detect contradictions if present
4. Summarize similarities
5. Mention each paper name while comparing

IMPORTANT:
- Do NOT assume there is only one paper
- Compare paper-wise
- Give meaningful technical comparison
- Avoid generic answers

Context:
{comparison_context}

Output format:
📌 Paper-wise Summary:
📌 Key Differences:
📌 Contradictions:
📌 Similarities:
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
                print("API KEY FOUND:", groq_api_key is not None)

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

                response_json = response.json()
                print("Groq Response:", response_json)

                if "choices" in response_json:
                    comparison_answer = response_json["choices"][0]["message"]["content"]
                else:
                    comparison_answer = f"Groq API Error: {response_json}"

        with col2:
            st.markdown(f"""
            <div style='background-color:#1e222b;padding:15px;border-radius:10px;margin-top:20px;'>
            📊 <b>Paper Comparison & Contradictions</b><br><br>
            {comparison_answer}
            </div>
            """, unsafe_allow_html=True)

    # STEP 6: Generate Novel Research Suggestions
    if suggest_button:
        with st.spinner("💡 Generating novel research ideas..."):

            suggestion_context = ""

            for paper_name, paper_chunk_list in paper_chunks.items():
                suggestion_context += f"\n\n===== PAPER: {paper_name} =====\n"
                suggestion_context += "\n".join(paper_chunk_list[:4])

            suggestion_prompt = f"""
You are an AI research innovation assistant.

You are given multiple research papers.
Your task is to:
1. Understand the core idea of each paper
2. Combine concepts creatively
3. Suggest novel hybrid research ideas
4. Propose possible future improvements
5. Mention how one paper’s methodology can enhance another

IMPORTANT:
- Be technically meaningful
- Avoid generic ideas
- Focus on AI/ML system innovation
- Mention practical applications

Context:
{suggestion_context}

Output format:
💡 Combined Research Ideas:
🚀 Possible Innovations:
🔬 Future Research Directions:
⚡ Practical Applications:
"""

            if mode == "Offline (Ollama)":
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3",
                        "prompt": suggestion_prompt,
                        "stream": False
                    }
                )
                suggestion_answer = response.json()["response"]

            else:
                groq_api_key = os.getenv("GROQ_API_KEY")
                print("API KEY FOUND:", groq_api_key is not None)

                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama3-8b-8192",
                        "messages": [
                            {"role": "user", "content": suggestion_prompt}
                        ]
                    }
                )

                response_json = response.json()
                print("Groq Response:", response_json)

                if "choices" in response_json:
                    suggestion_answer = response_json["choices"][0]["message"]["content"]
                else:
                    suggestion_answer = f"Groq API Error: {response_json}"

        st.markdown("---")
        st.subheader("💡 AI-Generated Research Suggestions")

        st.markdown(f"""
        <div style='background-color:#1e222b;padding:20px;border-radius:12px;margin-top:10px;'>
        {suggestion_answer}
        </div>
        """, unsafe_allow_html=True)

    # STEP 4: Show Source Context at Bottom
    if ask_button and question:
        st.markdown("---")
        st.subheader("📚 Retrieved Source Context")

        with st.expander("View Retrieved Chunks"):
            for i, (chunk, source) in enumerate(relevant_chunks):
                st.markdown(f"**Chunk {i+1} ({source}):**")
                st.write(chunk)
                st.markdown("---")