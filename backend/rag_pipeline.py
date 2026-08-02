import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
from paper_manager import (
    get_uploaded_papers,
    resolve_paper_path,
    BASE_DIR
)

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
INDEX_FAISS_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")

# --------------------------------------------------------
# Build Vector Store
# --------------------------------------------------------

def rebuild_vectorstore():
    papers = get_uploaded_papers()
    documents = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=250
    )

    for paper in papers:
        abs_path = resolve_paper_path(paper["path"])

        # Skip papers that no longer exist
        if not os.path.exists(abs_path):
            continue

        try:
            loader = PyPDFLoader(abs_path)
            pdf_docs = loader.load()
            chunks = splitter.split_documents(pdf_docs)

            for chunk in chunks:
                chunk.metadata["paper_id"] = paper["paper_id"]
                chunk.metadata["source"] = paper["name"]

            documents.extend(chunks)
        except Exception as e:
            print(f"Error reading PDF {abs_path}: {e}")

    # If no valid documents/papers remain
    if not documents:
        if os.path.exists(INDEX_FAISS_PATH):
            try:
                os.remove(INDEX_FAISS_PATH)
            except OSError:
                pass
        pkl_path = os.path.join(VECTORSTORE_DIR, "index.pkl")
        if os.path.exists(pkl_path):
            try:
                os.remove(pkl_path)
            except OSError:
                pass
        return

    vectorstore = FAISS.from_documents(
        documents,
        embedding_model
    )

    vectorstore.save_local(VECTORSTORE_DIR)

# --------------------------------------------------------
# Upload Paper
# --------------------------------------------------------

def process_pdf(pdf_path):
    rebuild_vectorstore()
    if not os.path.exists(INDEX_FAISS_PATH):
        return 0
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectorstore.index.ntotal

def ask_question(question):
    if not os.path.exists(INDEX_FAISS_PATH):
        return "No research papers uploaded yet."

    try:
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embedding_model,
            allow_dangerous_deserialization=True
        )


        import re

        paper_match = re.search(
            r"paper\s+(\d+)",
            question.lower()
        )

        if paper_match:

            requested_paper = int(
                paper_match.group(1)
            )

            docs = vectorstore.similarity_search(
                "",
                k=100
            )

            docs = [
                doc for doc in docs
                if doc.metadata.get("paper_id")
                == requested_paper
            ]

        else:

            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 12,
                    "fetch_k": 20
                }
            )

            docs = retriever.invoke(question)

        if not docs:
            return "No relevant content found in uploaded papers."

        context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                f"Content: {doc.page_content}"
                for doc in docs
            ]
        )

        prompt = f"""
You are an advanced AI research assistant.

Carefully analyze the provided research paper context and generate a clean, highly structured, well-formatted technical response.

CRITICAL FORMATTING INSTRUCTIONS:
- Do NOT output a single dense paragraph or wall of text.
- Divide your response into distinct, organized sections using Markdown subheadings (e.g. `### 📌 Section Title`).
- Present every key point or dataset feature as a bullet point (`- `) with bolded terms (`**Term Name**`).
- Add blank lines between sections and bullet list items for clean visual spacing.
- Clearly mention which paper the information comes from (e.g. `📘 **Source Paper:** ...`).
- If details are incomplete in the context, explicitly state: `*Note: Specific details not provided in the context.*`

Research Context:
{context}

User Question:
{question}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error during retrieval: {str(e)}"


def compare_papers():

    if not os.path.exists(INDEX_FAISS_PATH):
        return "No papers uploaded."

    try:

        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 14,
                "fetch_k": 24
            }
        )

        docs = retriever.invoke(
            "methodology architecture workflow tools limitations dataset results comparison"
        )

        context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                f"Content: {doc.page_content}"
                for doc in docs
            ]
        )

        prompt = f"""
You are an advanced AI research analyst.

Analyze and compare the uploaded research papers. Generate a clean, highly structured, well-formatted response.

CRITICAL FORMATTING INSTRUCTIONS:
- Do NOT output a single dense paragraph or wall of text.
- Divide your response into distinct, organized sections using Markdown subheadings (e.g. `### 📌 Section Title`).
- Present every key point as a bullet point (`- `) with bolded terms (`**Term Name**`).
- Add blank lines between sections and bullet list items for clean visual spacing.
- Clearly mention which paper the information comes from (e.g. `📘 **Paper:** ...`).
- If details are incomplete in the context, explicitly state: `*Note: Not explicitly mentioned in the paper.*`

Your comparison MUST include these sections as `### 📌` headings:
1. Research Objective
2. Methodology Differences
3. Workflow / System Architecture
4. AI Models or Algorithms Used
5. Tools / Frameworks / Libraries Used
6. Dataset Differences
7. Advantages of Each Paper
8. Limitations of Each Paper
9. Performance Comparison
10. Real-world Applications
11. Contradictions or Conflicting Claims
12. Final Technical Conclusion

Research Papers Context:
{context}

Rules:
- Give concise but technical answers.
- Clearly separate papers using `📘 **Paper Name**` labels.
- Use bullet points (`- `) whenever possible.
- Avoid repeating the same information.
- If no contradictions exist, explicitly say: `*No significant contradictions found between the papers.*`

Generate a professional research comparison report.
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Comparison Error: {str(e)}"


def generate_research_ideas():

    if not os.path.exists(INDEX_FAISS_PATH):
        return "No papers uploaded."

    try:

        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 12,
                "fetch_k": 20
            }
        )

        docs = retriever.invoke(
            "future work limitations architecture innovation hybrid approach"
        )

        context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                f"Content: {doc.page_content}"
                for doc in docs
            ]
        )

        prompt = f"""
You are an advanced AI research innovator.

Using the uploaded research papers, generate novel and technically meaningful research ideas.

CRITICAL FORMATTING INSTRUCTIONS:
- Do NOT output a single dense paragraph or wall of text.
- Structure each research idea as its own section with a Markdown subheading (e.g. `### 💡 Idea 1: Title Here`).
- Within each idea, use these sub-sections with bold labels:
  - **🔬 Problem Statement:** ...
  - **🚀 Proposed Innovation:** ...
  - **🛠️ Possible Tech Stack:** ...
  - **✅ Expected Advantages:** ...
  - **🌍 Real-world Applications:** ...
- Present every key point as a bullet point (`- `) with bolded terms (`**Term Name**`).
- Add blank lines between sections for clean visual spacing.
- Reference the source papers using `📘 **Paper:** ...` where relevant.

Focus on:
- hybrid AI systems
- unexplored improvements
- architecture optimizations
- real-world deployment ideas
- reducing limitations from existing papers
- combining methodologies creatively

Research Context:
{context}

Generate at least 3 strong research ideas.
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Idea Generation Error: {str(e)}"


# New function: recommend_related_papers
def recommend_related_papers():

    if not os.path.exists(INDEX_FAISS_PATH):
        return [
            {
                "title": "No papers uploaded",
                "reason": "Upload papers first to get recommendations.",
                "url": ""
            }
        ]

    try:

        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 20
            }
        )

        docs = retriever.invoke(
            "research topic abstract methodology application domain"
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are a research discovery assistant.

Analyze the uploaded research papers and recommend 5 closely related research papers.

CRITICAL FORMATTING INSTRUCTIONS:
- Do NOT output a single dense paragraph or wall of text.
- Structure each recommendation as its own section with a Markdown subheading (e.g. `### 📄 Recommendation 1`).
- Within each recommendation, use these bold labels on separate lines:
  - **📌 Title:** ...
  - **🔗 Why It's Relevant:** ...
  - **🔍 Search Phrase:** ...
- Add blank lines between recommendations for clean visual spacing.
- Reference the source papers from the context using `📘 **Related to:**` where relevant.

Research Context:
{context}

Generate exactly 5 recommendations.
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5
        )

        answer = response.choices[0].message.content

        return [
            {
                "title": "AI Recommended Papers",
                "reason": answer,
                "url": ""
            }
        ]

    except Exception as e:
        return [
            {
                "title": "Recommendation Error",
                "reason": str(e),
                "url": ""
            }
        ]