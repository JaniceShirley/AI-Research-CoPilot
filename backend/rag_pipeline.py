from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq
from dotenv import load_dotenv

import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

paper_counter = 0


def process_pdf(pdf_path):

    global paper_counter

    paper_counter += 1
    paper_id = paper_counter

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=250
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source"] = pdf_path
        chunk.metadata["paper_id"] = paper_id

    new_vectorstore = FAISS.from_documents(
        chunks,
        embedding_model
    )

    if os.path.exists("vectorstore/index.faiss"):

        old_vectorstore = FAISS.load_local(
            "vectorstore",
            embedding_model,
            allow_dangerous_deserialization=True
        )

        old_vectorstore.merge_from(new_vectorstore)

        old_vectorstore.save_local("vectorstore")

    else:

        new_vectorstore.save_local("vectorstore")

    return len(chunks)


def ask_question(question):

    if not os.path.exists("vectorstore/index.faiss"):
        return "No research papers uploaded yet."

    try:

        vectorstore = FAISS.load_local(
            "vectorstore",
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

Carefully analyze the provided research paper context and answer the user's question accurately.

Instructions:
- Use information from the uploaded papers.
- If multiple papers are involved, clearly mention which paper the answer comes from.
- If the answer is incomplete in the retrieved context, say so explicitly.
- Avoid vague statements.
- Give structured technical explanations whenever possible.
- For workflow/system questions, explain the pipeline step-by-step.
- Use bullet points for clarity.
- If relevant, compare methodologies, architectures, datasets, and applications.

Research Context:
{context}

Question:
{question}

Generate a clean, technical, research-style answer.
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

    if not os.path.exists("vectorstore/index.faiss"):
        return "No papers uploaded."

    try:

        vectorstore = FAISS.load_local(
            "vectorstore",
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

Analyze and compare the uploaded research papers in a highly structured format.

Your comparison MUST include:

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
- Clearly separate Paper 1 and Paper 2.
- Use bullet points whenever possible.
- Avoid repeating the same information.
- If some information is missing, say "Not explicitly mentioned in the paper".
- If no contradictions exist, explicitly say: "No significant contradictions found between the papers.".
- Highlight conflicting claims only if they are technically meaningful.

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

    if not os.path.exists("vectorstore/index.faiss"):
        return "No papers uploaded."

    try:

        vectorstore = FAISS.load_local(
            "vectorstore",
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

Focus on:
- hybrid AI systems
- unexplored improvements
- architecture optimizations
- real-world deployment ideas
- reducing limitations from existing papers
- combining methodologies creatively

Research Context:
{context}

Generate:
1. Research Idea Title
2. Problem Statement
3. Proposed Innovation
4. Possible Tech Stack
5. Expected Advantages
6. Real-world Applications

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