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

        def classify_query(q: str):
            """Deterministic lightweight classifier that routes queries into:
            - 'PAPER_METADATA'
            - 'PAPER_CONTENT'
            - 'CONVERSATIONAL'
            - 'OUT_OF_SCOPE'

            The classifier prefers PAPER_CONTENT for ambiguous cases so that
            questions that likely refer to the uploaded papers are not
            accidentally classified as out-of-scope.
            """
            s = (q or "").lower().strip()

            # Simple conversational patterns (greetings / assistant identity)
            conv_exact = [
                "hi", "hello", "hey", "hey there",
                "are you human", "are you a human", "who are you",
                "what can you do", "what is your name", "who am i",
                "how are you"
            ]
            for p in conv_exact:
                if s == p or s.startswith(p + " ") or s.startswith(p + "?"):
                    return "CONVERSATIONAL"

            if re.search(r"\b(hi|hello|hey)\b", s):
                return "CONVERSATIONAL"
            if re.search(r"\bare you\b.*\bhuman\b", s) or re.search(r"\bwho are you\b", s):
                return "CONVERSATIONAL"

            # Metadata keywords (explicit)
            metadata_keywords = [
                "author", "authors", "writer", "written by",
                "title", "affiliation", "institution", "publication",
                "journal", "conference", "doi", "year"
            ]
            if any(k in s for k in metadata_keywords):
                return "PAPER_METADATA"

            # Strong out-of-scope indicators (common general-knowledge questions)
            out_of_scope_keywords = [
                "capital of", "currency", "population", "president", "prime minister",
                "how to cook", "recipe", "what is the capital", "how far", "distance",
            ]
            # Only treat as OUT_OF_SCOPE if there's no hint it's about uploaded papers
            if any(k in s for k in out_of_scope_keywords) and "paper" not in s and not any(k in s for k in metadata_keywords):
                return "OUT_OF_SCOPE"

            # Content keywords (paper content / technical questions)
            content_keywords = [
                "method", "methodology", "dataset", "datasets", "results", "limitations",
                "conclusion", "conclusions", "architecture", "model", "models", "algorithm",
                "training", "evaluate", "evaluation", "experiment", "experiments", "approach",
                "why did they", "why did they use", "why did they choose", "what could be improved",
                "explain this", "what does paper", "summarize", "interpret", "compare", "comparison",
                "what does paper", "what does paper 2 say"
            ]
            if any(k in s for k in content_keywords):
                return "PAPER_CONTENT"

            # If user references a paper explicitly (paper 1, paper 2), treat as content unless it's metadata
            if re.search(r"paper\s+\d+", s):
                return "PAPER_CONTENT"

            # Pronoun-based heuristic: short questions that use "they/this/that/these/it" often refer to the loaded papers
            if re.search(r"\b(they|this|that|these|it|its|we)\b", s):
                return "PAPER_CONTENT"

            # Default: favor PAPER_CONTENT (conservative: avoid marking as out-of-scope)
            return "PAPER_CONTENT"

        question_lower = question.lower()

        # Detect explicit paper reference like "paper 2"
        paper_match = re.search(r"paper\s+(\d+)", question_lower)

        # Fetch uploaded papers list to allow safe paper selection
        papers = get_uploaded_papers()

        selected_paper = None
        if paper_match:
            requested_paper = int(paper_match.group(1))
            for p in papers:
                if p.get("paper_id") == requested_paper:
                    selected_paper = p
                    break

        # Classify the incoming question to decide routing
        route = classify_query(question)

        # Handle conversational and out-of-scope locally without querying FAISS
        if route == "CONVERSATIONAL":
            # Short, friendly assistant replies for conversational prompts
            if re.search(r"\bare you\b.*\bhuman\b", question_lower) or re.search(r"\bare you a human\b", question_lower):
                return "No, I'm an AI Research CoPilot designed to help you analyze and understand research papers."
            if re.search(r"\bwho are you\b", question_lower):
                return "I'm an AI Research CoPilot — I help you extract, summarize, and analyze uploaded research papers."
            if re.search(r"\bhi\b|\bhello\b|\bhey\b", question_lower):
                return "Hello — I'm an AI Research CoPilot. Upload papers and ask me about their metadata or content."
            return "I'm an AI Research CoPilot. I can help analyze uploaded research papers, extract metadata, compare papers, and generate ideas."

        if route == "OUT_OF_SCOPE":
            return "That question isn't related to the uploaded research papers. I can help you analyze, summarize, compare, or discuss your uploaded papers."

        # From here on, the question is either PAPER_METADATA or PAPER_CONTENT and requires the vectorstore
        if not os.path.exists(INDEX_FAISS_PATH):
            return "No research papers uploaded yet."

        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        # If it's a metadata-style question, prioritise first-page context
        if route == "PAPER_METADATA":
            # Determine target papers
            if selected_paper:
                target_papers = [selected_paper]
                ambiguous = False
            else:
                if len(papers) == 1:
                    target_papers = papers
                    ambiguous = False
                else:
                    # Multiple papers uploaded and user didn't specify — mark ambiguous
                    target_papers = papers
                    ambiguous = True

            # Collect first-page text for each target paper (fail gracefully if none)
            metadata_contexts = []
            for paper in target_papers:
                abs_path = resolve_paper_path(paper["path"])
                first_page_text = ""
                try:
                    loader = PyPDFLoader(abs_path)
                    pdf_docs = loader.load()
                    if pdf_docs and len(pdf_docs) >= 1:
                        first_page_text = pdf_docs[0].page_content or ""
                except Exception:
                    first_page_text = ""

                if first_page_text and first_page_text.strip():
                    metadata_contexts.append({
                        "paper": paper,
                        "first_page": first_page_text.strip()
                    })

            if not metadata_contexts:
                return "No extractable first-page text found for the uploaded paper(s); cannot determine metadata reliably."

            # Build a combined context that places first-page metadata at the top
            context = ""
            for item in metadata_contexts:
                paper = item["paper"]
                context += f"\n\n===== PAPER: {paper['name']} (page 1 metadata) =====\n"
                context += item["first_page"] + "\n"

            # Also include a small set of semantically relevant content chunks to supplement metadata
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,
                    "fetch_k": 12
                }
            )

            try:
                supplementary_docs = retriever.invoke(question)
            except Exception:
                supplementary_docs = []

            used_sources = set()
            for doc in supplementary_docs:
                src = doc.metadata.get("source", "Unknown")
                if src not in used_sources:
                    context += f"\n\n===== {src} =====\n"
                    used_sources.add(src)
                context += doc.page_content + "\n"

            # If ambiguous (multiple papers and user didn't pick), instruct the LLM to list authors per paper
            if ambiguous:
                clarification_note = (
                    "Note: Multiple papers are uploaded and the question did not identify a specific paper.\n"
                    "Please list authors and affiliations for each uploaded paper separately, and do NOT attribute authors to the wrong paper.\n"
                )
            else:
                clarification_note = ""

            prompt = f"""
You are an advanced AI research assistant focused on extracting factual paper metadata when available.

Important:
- Use ONLY the provided paper context. Do NOT invent or hallucinate author names or affiliations.
- If a paper's first page is image-only or missing text, state that you could not extract metadata instead of guessing.
- If multiple papers are present and the user did not specify which one, list each paper separately and indicate the paper source for every metadata item.

{clarification_note}

Context:
{context}

Question:
{question}

Answer:
"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            return response.choices[0].message.content

        # PAPER_CONTENT: Non-metadata / normal content questions: use existing retrieval flow
        if selected_paper:
            # If user asked about a specific paper, prefer chunks from that paper
            docs = vectorstore.similarity_search("", k=100)
            docs = [doc for doc in docs if doc.metadata.get("paper_id") == selected_paper.get("paper_id")]
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

        context = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
            for doc in docs
        ])

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