import os
import requests
import time
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

GROQ_MODEL = "groq/compound"

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
            - 'POTENTIAL_CONTENT' (needs relevance gating)

            The classifier only marks obvious conversational, metadata, or
            explicit content queries. Ambiguous/unknown queries are returned
            as 'POTENTIAL_CONTENT' so they can be checked against the FAISS
            index for actual relevance before being routed to PAPER_CONTENT
            or OUT_OF_SCOPE.
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

            # Explicit content keywords (clearly paper-related technical questions)
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

            # If user references a paper explicitly (paper 1, paper 2), treat as content
            if re.search(r"paper\s+\d+", s):
                return "PAPER_CONTENT"

            # Pronoun-based heuristic: short questions that use "they/this/that/these/it" may refer to the papers
            if re.search(r"\b(they|this|that|these|it|its|we)\b", s):
                return "POTENTIAL_CONTENT"

            # For ambiguous/unknown queries, don't assume PAPER_CONTENT — use the relevance gate
            return "POTENTIAL_CONTENT"

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
        print(f"Query route: {route}")

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

        # From here on, the question may require the vectorstore (PAPER_METADATA, PAPER_CONTENT, or POTENTIAL_CONTENT)
        if not os.path.exists(INDEX_FAISS_PATH):
            return "No research papers uploaded yet."

        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        # PAPER_METADATA: existing metadata logic (unchanged)
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
            for doc in supplementary_docs[:6]:
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
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            return response.choices[0].message.content

        # For routes that require evaluating content relevance (PAPER_CONTENT or POTENTIAL_CONTENT)
        # Implement a retrieval relevance gate using FAISS scores before constructing the final prompt.
        def relevance_gate(query, selected_paper=None, top_k=6, score_k=10):
            """Return (is_relevant: bool, top_docs: list, best_score: float)

            Uses vectorstore.similarity_search_with_score if available and falls back
            to retrieving candidate docs. Interprets score semantics heuristically:
            - If score values appear to be distances (max_score > 1.5), lower is better.
              Use DISTANCE_THRESHOLD (lower => more similar).
            - Otherwise treat scores as similarity (higher is better) and use SIMILARITY_THRESHOLD.

            Thresholds are chosen conservatively: require good evidence of relevance
            (median of top-3 within threshold) before declaring relevant.
            """
            try:
                results = vectorstore.similarity_search_with_score(query, k=score_k)
                # results is a list of (Document, score)
            except Exception:
                # Fall back to MMR retriever if method unavailable
                retr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "fetch_k": score_k})
                try:
                    candidates = retr.invoke(query)
                    # emulate scores by asking vectorstore to compute distances for these docs is difficult here
                    # treat as relevant conservatively
                    return True, candidates[:top_k], None
                except Exception:
                    return False, [], None

            if not results:
                return False, [], None

            # Optionally filter results to the selected_paper if provided
            if selected_paper is not None:
                paper_id = selected_paper.get("paper_id")
                results = [(doc, score) for doc, score in results if doc.metadata.get("paper_id") == paper_id]
                if not results:
                    # No candidates from the specified paper
                    return False, [], None

            # Sort by score ascending (assume lower better) but we'll detect semantics
            results_sorted = sorted(results, key=lambda x: x[1])
            scores = [s for _, s in results_sorted]

            best_score = scores[0]
            # Inspect score range to guess semantics
            max_score = max(scores)
            min_score = min(scores)

            # Heuristics to decide if score is distance (lower better) or similarity (higher better)
            if max_score > 1.5:
                # Treat as distance (lower better)
                DISTANCE_THRESHOLD = 0.9
                # take median of top-3
                top3 = scores[:3]
                median_top3 = sorted(top3)[len(top3)//2]
                is_relevant = median_top3 <= DISTANCE_THRESHOLD
                print(f"Best relevance (distance) = {best_score}, median_top3={median_top3}, threshold={DISTANCE_THRESHOLD}")
                top_docs = [d for d, _ in results_sorted[:top_k]]
                return is_relevant, top_docs, best_score
            else:
                # Treat as similarity (higher better)
                SIMILARITY_THRESHOLD = 0.20
                # take median of top-3 (largest scores)
                top3 = scores[-3:]
                median_top3 = sorted(top3)[len(top3)//2]
                is_relevant = median_top3 >= SIMILARITY_THRESHOLD
                print(f"Best relevance (similarity) = {best_score}, median_top3={median_top3}, threshold={SIMILARITY_THRESHOLD}")
                # get corresponding docs for top similarity
                results_by_sim = sorted(results, key=lambda x: x[1], reverse=True)
                top_docs = [d for d, _ in results_by_sim[:top_k]]
                return is_relevant, top_docs, best_score

        # Decide path for POTENTIAL_CONTENT: do a relevance gate
        if route == "POTENTIAL_CONTENT":
            is_relevant, candidate_docs, best_score = relevance_gate(question, selected_paper=selected_paper, top_k=6, score_k=20)
            print(f"Query route after gate: {'PAPER_CONTENT' if is_relevant else 'OUT_OF_SCOPE'}; Best score: {best_score}")
            if not is_relevant:
                return "That question doesn't appear to be related to the uploaded research papers. Please ask a question about the paper."
            # proceed with PAPER_CONTENT flow using candidate_docs
            docs = candidate_docs
        elif route == "PAPER_CONTENT":
            # Explicit content queries: retrieve as before
            if selected_paper:
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
            for doc in docs[:6]
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
            model=GROQ_MODEL,
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
                for doc in docs[:6]
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
            model=GROQ_MODEL,
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
                for doc in docs[:6]
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
            model=GROQ_MODEL,
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


# New function: recommend_related_papers using real Semantic Scholar API
def recommend_related_papers():

    if not os.path.exists(INDEX_FAISS_PATH):
        return [
            {
                "title": "No papers uploaded",
                "authors": "",
                "year": "",
                "venue": "",
                "relevance": "Upload papers first to get recommendations.",
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
            "research topic title methodology abstract application domain keywords"
        )

        context = "\n\n".join([doc.page_content for doc in docs[:6]])

        # 1. Extract concise 3-5 search terms using Groq
        search_query = "smart parking system IoT license plate recognition"
        try:
            prompt = f"""
Given the following excerpts from uploaded research paper(s), extract a concise 3 to 5 word search query suitable for searching related academic papers on Semantic Scholar.
Focus on the main research topic, methodology, or application domain.
Output ONLY the query keywords without punctuation or preamble.

Excerpts:
{context[:2000]}
"""
            kw_response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            extracted = kw_response.choices[0].message.content.strip().strip('"').strip("'")
            if extracted and len(extracted) > 3:
                search_query = extracted
        except Exception as e:
            print(f"Keyword extraction warning: {e}")

        # 2. Search Semantic Scholar Graph API
        ss_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        headers = {
            "User-Agent": "AI-Research-CoPilot/1.0 (contact: research@example.com)"
        }
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key

        params = {
            "query": search_query,
            "limit": 10,
            "fields": "title,authors,year,venue,externalIds,url,abstract,publicationVenue,openAccessPdf"
        }

        resp = None
        for attempt in range(2):
            try:
                resp = requests.get(ss_url, params=params, headers=headers, timeout=10)
                if resp.status_code == 200:
                    break
                elif resp.status_code == 429 and attempt == 0:
                    time.sleep(2)
            except Exception as req_err:
                print(f"Semantic Scholar request attempt {attempt} failed: {req_err}")

        print(f"[SemanticScholar] Query sent: {search_query}")
        print(f"[SemanticScholar] HTTP Status Code: {resp.status_code if resp else 'No Response'}")
        if resp and resp.status_code != 200:
            print(f"[SemanticScholar] Error Body: {resp.text[:300]}")

        if not resp or resp.status_code != 200:
            return [
                {
                    "error": "Unable to fetch related papers right now. Please try again.",
                    "title": "Unable to fetch related papers right now. Please try again.",
                    "authors": "",
                    "year": "",
                    "venue": "",
                    "relevance": "Unable to fetch related papers right now. Please try again.",
                    "reason": "Unable to fetch related papers right now. Please try again.",
                    "url": ""
                }
            ]

        data = resp.json()
        raw_papers = data.get("data", [])
        print(f"[SemanticScholar] Results returned: {len(raw_papers)}")
        if not raw_papers:
            return []

        results = []
        for p in raw_papers:
            title = p.get("title")
            if not title:
                continue

            # Authors
            authors_list = p.get("authors") or []
            if authors_list:
                authors_str = ", ".join([a.get("name", "") for a in authors_list if a.get("name")])
            else:
                authors_str = "Unknown Authors"

            # Year
            year_val = p.get("year")
            year_str = str(year_val) if year_val else "N/A"

            # Venue / Journal
            venue = p.get("venue")
            if not venue and p.get("publicationVenue"):
                pub_v = p.get("publicationVenue")
                if isinstance(pub_v, dict):
                    venue = pub_v.get("name")
            if not venue:
                venue = "Academic Venue"

            # URL extraction logic
            paper_url = p.get("url")
            if not paper_url and p.get("openAccessPdf"):
                pdf_info = p.get("openAccessPdf")
                if isinstance(pdf_info, dict):
                    paper_url = pdf_info.get("url")
            if not paper_url and p.get("externalIds"):
                ext = p.get("externalIds")
                if isinstance(ext, dict):
                    if "DOI" in ext:
                        paper_url = f"https://doi.org/{ext['DOI']}"
                    elif "ArXiv" in ext:
                        paper_url = f"https://arxiv.org/abs/{ext['ArXiv']}"
            if not paper_url and p.get("paperId"):
                paper_url = f"https://www.semanticscholar.org/paper/{p['paperId']}"

            # Skip paper if no usable URL is available
            if not paper_url:
                continue

            # Relevance sentence
            abstract = p.get("abstract") or ""
            relevance = f"Explores {search_query} and related methodologies, making it closely related to the uploaded research."
            if abstract:
                try:
                    rel_prompt = f"""
Given the topic '{search_query}', summarize in exactly ONE concise line why this paper is related to the uploaded research:
Paper Title: {title}
Abstract Snippet: {abstract[:300]}

Requirement: Output ONE single sentence explaining why it is relevant. Do not add labels or quotes.
"""
                    rel_resp = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[{"role": "user", "content": rel_prompt}],
                        temperature=0.3,
                        max_tokens=60
                    )
                    rel_text = rel_resp.choices[0].message.content.strip().replace("\n", " ")
                    if rel_text:
                        relevance = rel_text
                except Exception:
                    pass

            results.append({
                "title": title,
                "authors": authors_str,
                "year": year_str,
                "venue": venue,
                "relevance": relevance,
                "url": paper_url
            })

            if len(results) == 5:
                break

        return results

    except Exception as e:
        print(f"Error in recommend_related_papers: {e}")
        return [
            {
                "error": "Unable to fetch related papers right now. Please try again.",
                "title": "Unable to fetch related papers right now. Please try again.",
                "authors": "",
                "year": "",
                "venue": "",
                "relevance": "Unable to fetch related papers right now. Please try again.",
                "reason": "Unable to fetch related papers right now. Please try again.",
                "url": ""
            }
        ]