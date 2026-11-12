# RAG evaluation

A reproducible, offline evaluation of the AI Research Co-Pilot RAG pipeline
(`backend/rag_pipeline.py`). It sends dataset questions through the
**unmodified** production `ask_question`, then measures:

- whether the right chunks were retrieved
- whether answers stay faithful to that context
- whether answers address the question
- how long each stage takes

Results are written to JSON, CSV and a Markdown report.

The framework lives entirely in `evaluation/`. The FastAPI app does not
import it, and it changes no file in `backend/` or `frontend/`.

## How it works

```
dataset.json ──► isolated temp workspace ──► production ingestion (register_uploaded_paper + process_pdf)
                     │                         └─► FAISS index for a fresh eval-<random> session
                     ▼
     for each question: rag_pipeline.ask_question(session, question)
        ├─ FAISS search methods wrapped  → retrieved chunks (in order) + retrieval time
        ├─ rag_pipeline.client wrapped   → final prompt + answer + generation time
        └─ stdout captured               → the pipeline's own "Query route: …" log
                     ▼
     metrics (deterministic + LLM judge) ─► aggregate ─► results.json / results.csv / report.md
```

**Isolation and safety**

- During a run, `paper_manager.BASE_DIR/METADATA_DIR/UPLOADS_BASE_DIR` and
  `rag_pipeline.BASE_DIR` point at a temporary directory. The original paths
  are restored afterwards, even on error.
- The source PDFs are only read and copied.
- Every run takes a snapshot of these folders before and after (path, size,
  mtime, sha256): `backend/uploads`, `backend/metadata`, `backend/vectorstore`
  and the papers folder. The report's **Data integrity** section shows the
  result.
- The evaluation uses its own random session ID, so per-session isolation
  still applies inside the workspace.
- The FAISS and Groq wrappers are removed when the run ends. The embedding
  model, LLM, prompts, chunking and retrieval settings are the production
  ones.

## 1. Prepare the dataset

The dataset is a JSON file. `datasets/template.json` is an empty template, and
`datasets/parkease_papers.json` is a filled-in example for the two papers
previously uploaded to the app.

```json
{
  "name": "my-dataset-v1",
  "status": "draft",
  "papers": [{"file": "my_paper.pdf", "sha256": "<optional>"}],
  "items": [{
    "id": "unique-id",
    "question": "A question a user would ask",
    "question_type": "content | metadata | comparison",
    "expected_answer": "The correct answer, taken from the paper",
    "key_facts": ["93.7%", "YOLOv12"],
    "relevant_sources": [
      {"paper": "my_paper.pdf", "page": 4, "evidence": "verbatim quote from page 4"}
    ],
    "verified": false
  }]
}
```

What you need to provide for each item:

| Field | Required | Notes |
|---|---|---|
| `question` | yes | Phrase it the way a user would type it into the app. Routing depends on wording (e.g. "author" sends it to the metadata path). |
| `expected_answer` | yes | Written from the paper. It is shown next to the generated answer for review; no metric depends on it. |
| `relevant_sources[].paper` | yes | Must match a file in `papers`. |
| `relevant_sources[].page` | recommended | 1-based PDF page, as shown in a PDF viewer. Leave it out to accept any page. |
| `relevant_sources[].evidence` | yes | A **verbatim** quote from the PDF, ideally under 250 characters. The index uses 500-character chunks with 250 overlap, so a short quote always fits inside one chunk. Copy it from the PDF text, not from memory. |
| `key_facts` | optional | Short strings a correct answer must contain (numbers, model names). Used for the supplementary key-fact coverage metric. |
| `verified` | — | Set to `true` once someone who knows the paper has checked the item. The report warns while any item is unverified. |
| `papers[].sha256` | optional | `shasum -a 256 file.pdf`. The report warns if the PDF you run against differs. |

Run `validate` after editing a dataset. It indexes the papers with the
production code (no LLM calls) and checks that every evidence quote appears in
at least one chunk. A quote that can't be found would score 0 recall no matter
how well retrieval works.

### About `parkease_papers.json`

It has 14 questions about *Automatic License Plate Recognition* (Chang et
al., IEEE T-ITS 2004) and *A Dual AI and IoT Approach to Smart Parking*
(ICDICI-2025).

- Every expected answer and evidence quote was taken from the text pypdf
  extracts from those exact PDFs (their checksums are in the file).
- Claude Code drafted it, so every item is `"verified": false` until you
  review it.
- The PDFs themselves are not in the repository. Pass their folder with
  `--papers-dir`.

## 2. Run the evaluation

Prerequisites:

- the backend's Python environment (`pip install -r backend/requirements.txt`)
- `GROQ_API_KEY` in a `.env` file at the repo root or in `backend/`, or passed
  with `--env-file`

The key is read from the environment and is never printed or written to
results. All commands below run from the repository root.

```bash
# Check the dataset against the real index (no LLM calls)
python -m evaluation validate --dataset evaluation/datasets/parkease_papers.json --papers-dir /path/to/pdfs
```

```bash
# Full evaluation
python -m evaluation run --dataset evaluation/datasets/parkease_papers.json --papers-dir /path/to/pdfs
```

Useful options for `run`:

| Option | Default | Meaning |
|---|---|---|
| `--k 1 3 5 10` | `1 3 5 10` | K values for Recall@K / Hit@K |
| `--judge-model M` | `openai/gpt-oss-120b` (or `$EVAL_JUDGE_MODEL`) | Groq model used as LLM judge |
| `--judge-max-tokens N` | unset | Cap judge output tokens. Needed when the judge's per-minute output-token limit is small (Groq on-demand `qwen/qwen3.8-27b`: 1000/min → use 900) |
| `--no-judge` | off | Deterministic metrics only (no judge calls) |
| `--generation-model M` | production `GROQ_MODEL` | **Evaluation only:** temporarily replace the answer model. The report states the override and the production model. Production code is not changed. |
| `--ids a b` / `--limit N` | all | Evaluate a subset |
| `--sleep S`, `--retries N`, `--retry-wait S` | 0, 2, 20 | Help with Groq rate limits |
| `--keep-workspace` | off | Keep the temporary uploads and FAISS index for inspection |
| `--verbose` | off | Echo the pipeline's own log lines |
| `--output-dir` | `evaluation/results/` (git-ignored) | Each run writes to `<timestamp>-<dataset-name>/` |

Exit codes: 0 = all questions answered without pipeline errors, 2 = some
pipeline errors (results are still written), 1 = dataset problem.

Tests (standard library `unittest`; no network, keys or models needed):

```bash
python -m unittest discover -s evaluation/tests -t .
```

The optional live smoke test calls the real app's upload, papers, ask,
compare, ideas, recommend and delete endpoints in an isolated workspace. It
makes real Groq calls and one Semantic Scholar call.

```bash
RAG_EVAL_LIVE=1 RAG_EVAL_PAPER=/path/to/paper.pdf python -m unittest evaluation.tests.test_live_app_smoke -v
```

## 3. Metrics

### A. Retrieval Recall@K (deterministic)

**The ranked list.** It contains every chunk the pipeline's FAISS calls
returned for the question, in the order FAISS returned them, with duplicates
removed. Which call that is depends on the route:

- `PAPER_CONTENT`: the MMR retriever (k=12, fetch_k=20).
- Relevance-gated questions: `similarity_search_with_score` (k=20).
- Metadata questions: MMR (k=6).

**Matching chunks to evidence.** FAISS chunk IDs change on every rebuild, so
items point to evidence quotes instead of chunk IDs. A chunk *covers* an
evidence item when all three hold:

- it comes from the same PDF;
- it comes from the same page, if a page is given;
- it contains at least `--match-threshold` (default 60%) of the quote's word
  trigrams, after lowercasing and stripping punctuation and PDF spacing
  artefacts.

**The scores.**

- **Recall@K** = evidence items covered by the top-K chunks ÷ all evidence
  items for that question. The reported value is the mean over questions.
  Questions without evidence are excluded rather than counted as 0 or 1.
- **Hit@K**: 1 if at least one evidence item is covered in the top K.
- **Context recall**: the same check over only the chunks that were actually
  pasted into the final LLM prompt (found by locating each chunk's text in the
  prompt). This shows what the model could see, which can differ from the
  FAISS ranking. For example, the pipeline uses only the first 6 of 12 MMR
  results, and the relevance gate re-sorts its results.

### B. Faithfulness (LLM-as-a-judge, labelled `faithfulness_llm_judge`)

The judge receives the context block of the final prompt (exactly what the
answer model saw) and the answer. It runs at temperature 0 and must return
JSON.

1. The judge splits the answer into atomic factual claims. It skips formatting
   and "not provided" notes.
2. It labels each claim **supported**, **unsupported** or **contradicted**,
   using only the context and not its own knowledge.
3. It quotes the supporting span for each supported claim.

**Faithfulness = supported ÷ all claims.** Unsupported claims count as
failures even if they are true in the real world, so nothing absent from the
context is treated as factual. Unknown labels are treated as unsupported. An
answer with no claims gets `n/a`.

Every claim and its verdict is saved in `results.json` for auditing.

Two transparent, non-LLM checks are reported next to it:

- **Unsupported numbers**: numbers in the answer that appear nowhere in the
  context. This is the most common kind of hallucination in result summaries.
  Integers 0–10 are ignored as list numbering.
- **Lexical support**: the share of the answer's content words that occur in
  the context. This is a rough proxy only, not a faithfulness score.

### C. Answer relevance (LLM-as-a-judge, labelled `answer_relevance_llm_judge`)

A 1–5 rubric scores whether the answer addresses the question. It does not
check correctness. The score is normalised to 0–1 as (score − 1) ÷ 4.

| Score | Meaning |
|---|---|
| 5 | Answers every part of the question |
| 4 | Main question answered; minor gaps or padding |
| 3 | Partial; important parts missing |
| 2 | Mostly off-target |
| 1 | Doesn't address it (refusal, wrong topic, error) |

Refusals are judged too, so refusing an in-scope question scores 1. The
judge's one-sentence reason is saved for each question.

Two supplementary, deterministic checks:

- **Question↔answer embedding cosine**: uses the production MiniLM model.
- **Key-fact coverage**: the share of the item's `key_facts` that appear in
  the answer.

### D. Latency

| Measure | Definition |
|---|---|
| Retrieval | Wall time inside the FAISS search calls, including query embedding. Summed if a question makes several calls. |
| Generation | Wall time inside the Groq `chat.completions.create` calls. |
| Total | Wall time of the whole `ask_question` call. |
| Other | Total − retrieval − generation. Covers loading the FAISS index from disk (done twice per question by the current code), routing, and PDF parsing for metadata questions. |

The report gives mean, median, p95 and n:

- Generation and total latency cover answered questions only.
- Retrieval latency covers every question where FAISS was queried, including
  those where generation later failed.
- Latency is measured in-process. It excludes HTTP and frontend time and
  varies with the machine and the Groq API load.

## 4. Interpreting results

- **Low Recall@K but high Hit@K**: retrieval finds part of the evidence.
  Multi-part questions need all their evidence.
- **Recall@10 much higher than Context recall**: FAISS finds the evidence, but
  it doesn't reach the prompt. Look at how many chunks are kept and how they
  are ordered.
- **Context recall high, faithfulness low**: the model has the evidence but
  adds unsupported details. Check `unsupported_numbers` and the claim list.
- **Faithfulness high, relevance low**: the answer sticks to the context but
  doesn't answer the question. This often means the wrong context was
  retrieved.
- **`pipeline_status`**:
  - `answered` means the LLM was called.
  - `no_generation` means the router or relevance gate answered or refused
    without calling the LLM.
  - `error` means the pipeline returned an `Error during retrieval: …` string.
    Errors are excluded from answer-quality and generation-latency averages
    and listed in the report.
- With 14 questions, one question moves an average by about 7 points. Treat
  differences smaller than that as noise, and compare runs on the same
  dataset version (the report records its sha256).

## 5. Limitations

- **LLM-as-a-judge is not ground truth.**
  - The judge can mis-split claims, be lenient with paraphrases, or miss
    subtle contradictions.
  - Scores can shift between judge models and model versions.
  - If the judge comes from the same model family as the generator, it may
    favour that model's answers.
  - Report which judge model you used (the report does this automatically),
    and spot-check the saved claims.
- **Judge context is capped** at 24,000 characters. `judge_context_truncated`
  marks answers where this applied.
- **Evidence matching is lexical.** A chunk that paraphrases the evidence
  doesn't count. A different passage with the same wording would count, but
  the paper and page filters make that unlikely.
- **Small datasets give high-variance averages.** The bundled dataset is a
  Claude-drafted, unreviewed starting point and covers only two papers.
- **Coverage is limited to `ask_question`**, which is the RAG path.
  `compare`, `ideas` and `recommend` produce open-ended text with no single
  correct answer and are only smoke-tested. Recommendations also depend on the
  external Semantic Scholar API.
- **Faithfulness is relative to the prompt context.** For metadata questions
  that includes the first page of each paper.
- **The FAISS index is built from scratch for each run**, using the same code
  and models as production. Results reflect the code at the recorded git
  commit.
