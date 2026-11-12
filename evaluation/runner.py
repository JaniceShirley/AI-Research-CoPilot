"""Evaluation pipeline: dataset -> production RAG -> metrics -> aggregate.

The heavy dependencies (the backend modules, the judge client, the embedding
function) are passed in, so tests can run the whole loop with fakes.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from . import metrics
from .instrumentation import PipelineInstrumentation, doc_to_chunk
from .judge import JudgeError, LLMJudge
from .matching import best_coverage
from .workspace import (diff_snapshots, file_sha256, ingest_papers, isolated_workspace,
                        new_eval_session_id, snapshot_tree)

PIPELINE_ERROR_PREFIXES = ("Error during retrieval:",)
_RATE_LIMIT_HINTS = ("rate limit", "rate_limit", "429", "too many requests",
                     "connection error", "timed out", "503", "502")  # transient: worth a retry


@dataclass
class RunConfig:
    k_values: Sequence[int] = (1, 3, 5, 10)
    match_threshold: float = 0.6
    retries: int = 2
    retry_wait_s: float = 20.0
    sleep_between_s: float = 0.0
    keep_workspace: bool = False
    echo_pipeline_logs: bool = False
    progress: Callable[[str], None] = print
    extra: dict = field(default_factory=dict)


def _pipeline_status(answer: str, trace) -> str:
    if any(answer.startswith(p) for p in PIPELINE_ERROR_PREFIXES):
        return "error"
    if any(c.error for c in trace.llm_calls):
        return "error"
    if trace.llm_calls:
        return "answered"
    return "no_generation"  # conversational / out-of-scope / gate refusal


def _is_rate_limited(answer: str) -> bool:
    low = answer.lower()
    return any(h in low for h in _RATE_LIMIT_HINTS)


def index_chunks(pipeline: Any, session_id: str) -> list:
    """Every chunk stored in the session's FAISS index (for evidence checks)."""
    vdir, faiss_path = pipeline.get_session_vectorstore_paths(session_id)
    if not os.path.exists(faiss_path):
        return []
    vs = pipeline.FAISS.load_local(vdir, pipeline.embedding_model,
                                   allow_dangerous_deserialization=True)
    return [doc_to_chunk(d) for d in vs.docstore._dict.values()]


def check_evidence(items: Sequence[dict], chunks: Sequence[dict], threshold: float) -> list:
    """For every evidence quote, the best coverage by any chunk in the index.

    A quote below ``threshold`` can never be "retrieved", which would make its
    Recall@K 0 for reasons unrelated to retrieval quality, so it is reported.
    """
    out = []
    for it in items:
        for j, src in enumerate(it.get("relevant_sources", [])):
            cov = best_coverage(src, chunks)
            out.append({"item_id": it["id"], "source_index": j, "paper": src.get("paper"),
                        "page": src.get("page"), "best_coverage": round(cov, 3),
                        "ok": cov >= threshold})
    return out


def evaluate_item(item: dict, pipeline: Any, instr: PipelineInstrumentation, session_id: str,
                  cfg: RunConfig, judge: Optional[LLMJudge] = None,
                  embed: Optional[Callable[[str], list]] = None) -> dict:
    question = item["question"]
    attempts = 0
    while True:
        attempts += 1
        with instr.capture(echo=cfg.echo_pipeline_logs) as trace:
            t0 = time.perf_counter()
            answer = pipeline.ask_question(session_id, question)
            total_s = time.perf_counter() - t0
        answer = answer if isinstance(answer, str) else str(answer)
        status = _pipeline_status(answer, trace)
        if status == "error" and _is_rate_limited(answer + " ".join(c.error or "" for c in trace.llm_calls)) \
                and attempts <= cfg.retries:
            cfg.progress(f"    rate limited; waiting {cfg.retry_wait_s:.0f}s and retrying")
            time.sleep(cfg.retry_wait_s)
            continue
        break

    relevant = item.get("relevant_sources", [])
    ranked = trace.ranked_chunks()
    ctx_chunks = trace.context_chunks()
    context = trace.answer_context()
    retrieval_s = trace.retrieval_seconds if trace.retrieval_calls else None
    generation_s = trace.generation_seconds if trace.llm_calls else None

    rec: dict = {
        "id": item["id"],
        "question": question,
        "question_type": item.get("question_type"),
        "expected_answer": item.get("expected_answer"),
        "answer": answer,
        "pipeline_status": status,
        "routes": trace.routes(),
        "attempts": attempts,
        "n_relevant_sources": len(relevant),
        "n_retrieved_chunks": len(ranked),
        "n_context_chunks": len(ctx_chunks),
        "latency_retrieval_s": retrieval_s,
        "latency_generation_s": generation_s,
        "latency_total_s": total_s,
        "latency_other_s": total_s - (retrieval_s or 0) - (generation_s or 0),
        "n_llm_calls": len(trace.llm_calls),
        "generation_model": trace.llm_calls[-1].model if trace.llm_calls else None,
        "retrieval_calls": [
            {"method": c.method, "kwargs": c.kwargs, "seconds": c.seconds,
             "n_results": len(c.chunks), "scores": c.scores}
            for c in trace.retrieval_calls
        ],
        "retrieved_chunks": [
            {"rank": i, "source": ch["source"], "page": ch["page"],
             "relevant": any(metrics.chunk_matches_source(s, ch, cfg.match_threshold) for s in relevant),
             "in_llm_context": ch in ctx_chunks, "text": ch["text"]}
            for i, ch in enumerate(ranked, start=1)
        ],
        "errors": [],
    }

    # A. Retrieval Recall@K (over FAISS's ranking) and context recall
    for k in cfg.k_values:
        rec[f"recall@{k}"] = metrics.recall_at_k(relevant, ranked, k, cfg.match_threshold)
        rec[f"hit@{k}"] = metrics.hit_at_k(relevant, ranked, k, cfg.match_threshold)
    rec["context_recall"] = (metrics.recall_at_k(relevant, ctx_chunks, max(len(ctx_chunks), 1),
                                                 cfg.match_threshold) if relevant else None)
    rec["first_relevant_rank"] = metrics.first_relevant_rank(relevant, ranked, cfg.match_threshold)

    # Supplementary deterministic checks
    answered = status == "answered"
    rec["lexical_support"] = metrics.lexical_support(answer, context) if answered else None
    rec["unsupported_numbers"] = metrics.unsupported_numbers(answer, context) if answered else []
    # Error strings are not answers: don't score them.
    rec["key_fact_coverage"] = (metrics.key_fact_coverage(answer, item.get("key_facts") or [])
                                if status != "error" else None)
    rec["embedding_relevance"] = None
    if embed is not None and status != "error":
        try:
            rec["embedding_relevance"] = metrics.cosine_similarity(embed(question), embed(answer))
        except Exception as e:
            rec["errors"].append(f"embedding_relevance: {e}")

    # B/C. LLM-as-a-judge
    rec["faithfulness_llm_judge"] = None
    rec["answer_relevance_llm_judge"] = None
    rec["answer_relevance_rubric"] = None
    rec["faithfulness_claims"] = []
    if judge is not None and answered:
        try:
            f = judge.faithfulness(context, answer)
            rec["faithfulness_llm_judge"] = f["score"]
            rec["faithfulness_claims"] = f["claims"]
            rec["n_claims"] = f["n_claims"]
            rec["n_unsupported_claims"] = f["n_unsupported"]
            rec["n_contradicted_claims"] = f["n_contradicted"]
            rec["judge_context_truncated"] = f["context_truncated"]
        except JudgeError as e:
            rec["errors"].append(f"faithfulness judge: {e}")
    if judge is not None and status != "error":
        # Relevance is judged for refusals too: a refusal to an in-scope
        # question is an irrelevant answer and should score 1/5.
        try:
            r = judge.relevance(question, answer)
            rec["answer_relevance_llm_judge"] = r["score"]
            rec["answer_relevance_rubric"] = r["rubric_score"]
            rec["answer_relevance_reason"] = r["reason"]
        except JudgeError as e:
            rec["errors"].append(f"relevance judge: {e}")
    return rec


def aggregate(records: Sequence[dict], k_values: Sequence[int]) -> dict:
    answered = [r for r in records if r["pipeline_status"] == "answered"]
    ran = [r for r in records if r["pipeline_status"] != "error"]
    out = {
        "n_questions": len(records),
        "n_answered": len(answered),
        "n_no_generation": sum(1 for r in records if r["pipeline_status"] == "no_generation"),
        "n_errors": sum(1 for r in records if r["pipeline_status"] == "error"),
        "routes": {},
    }
    for r in records:
        key = " -> ".join(r["routes"]) or "(none logged)"
        out["routes"][key] = out["routes"].get(key, 0) + 1
    for k in k_values:
        out[f"recall@{k}"] = metrics.summarize(r.get(f"recall@{k}") for r in records)
        out[f"hit@{k}"] = metrics.summarize(r.get(f"hit@{k}") for r in records)
    for name in ("context_recall", "faithfulness_llm_judge", "answer_relevance_llm_judge",
                 "lexical_support", "key_fact_coverage", "embedding_relevance"):
        out[name] = metrics.summarize(r.get(name) for r in records)
    out["latency_retrieval_s"] = metrics.summarize(r["latency_retrieval_s"] for r in records)
    out["latency_generation_s"] = metrics.summarize(r["latency_generation_s"] for r in answered)
    out["latency_total_s"] = metrics.summarize(r["latency_total_s"] for r in answered)
    out["latency_total_s_all_routes"] = metrics.summarize(r["latency_total_s"] for r in ran)
    out["n_answers_with_unsupported_numbers"] = sum(1 for r in answered if r.get("unsupported_numbers"))
    return out


def run_evaluation(dataset: dict, items: Sequence[dict], pdf_paths: Sequence[str],
                   paper_manager: Any, pipeline: Any, vectorstore_cls: Any, cfg: RunConfig,
                   judge: Optional[LLMJudge] = None,
                   embed: Optional[Callable[[str], list]] = None,
                   extra_protected_paths: Sequence[str] = ()) -> dict:
    """Full run inside an isolated workspace. Returns a JSON-serialisable result."""
    started = _dt.datetime.now(_dt.timezone.utc)
    real_base = paper_manager.BASE_DIR
    protected = [os.path.join(real_base, d) for d in ("uploads", "metadata", "vectorstore")]
    protected += [os.path.dirname(p) for p in pdf_paths] + list(extra_protected_paths)
    protected = sorted(set(protected))
    before = snapshot_tree(protected)

    paper_info = []
    for spec, path in zip(dataset["papers"], pdf_paths):
        sha = file_sha256(path)
        paper_info.append({"file": os.path.basename(path), "sha256": sha,
                           "expected_sha256": spec.get("sha256"),
                           "sha256_matches": (spec.get("sha256") in (None, "", sha))})

    records: list = []
    session_id = new_eval_session_id()
    with isolated_workspace(paper_manager, pipeline, keep=cfg.keep_workspace) as workspace:
        cfg.progress(f"Workspace: {workspace} (session {session_id})")
        cfg.progress(f"Indexing {len(pdf_paths)} paper(s) with the production ingestion code...")
        t0 = time.perf_counter()
        ingest = ingest_papers(paper_manager, pipeline, session_id, pdf_paths)
        index_s = time.perf_counter() - t0
        chunks = index_chunks(pipeline, session_id)
        evidence = check_evidence(items, chunks, cfg.match_threshold)
        bad = [e for e in evidence if not e["ok"]]
        cfg.progress(f"Indexed {len(chunks)} chunks in {index_s:.1f}s; "
                     f"{len(evidence) - len(bad)}/{len(evidence)} evidence quotes found in the index")

        with PipelineInstrumentation(pipeline, vectorstore_cls) as instr:
            for i, item in enumerate(items, start=1):
                cfg.progress(f"[{i}/{len(items)}] {item['id']}: {item['question'][:70]}")
                rec = evaluate_item(item, pipeline, instr, session_id, cfg, judge, embed)
                cfg.progress(f"    status={rec['pipeline_status']} route={'>'.join(rec['routes'])} "
                             f"total={rec['latency_total_s']:.2f}s")
                records.append(rec)
                if cfg.sleep_between_s and i < len(items):
                    time.sleep(cfg.sleep_between_s)

    after = snapshot_tree(protected)
    integrity = diff_snapshots(before, after)
    integrity["protected_paths"] = protected
    integrity["unchanged"] = not any(integrity[k] for k in ("added", "removed", "modified"))

    return {
        "run": {
            "started_utc": started.isoformat(timespec="seconds"),
            "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "session_id": session_id,
            "kept_workspace": workspace if cfg.keep_workspace else None,
            "index_build_seconds": index_s,
            "n_index_chunks": len(chunks),
            **cfg.extra,
        },
        "dataset": {
            "name": dataset.get("name"),
            "path": dataset.get("_path"),
            "sha256": dataset.get("_sha256"),
            "status": dataset.get("status"),
            "description": dataset.get("description"),
            "n_items_total": len(dataset["items"]),
            "n_items_evaluated": len(items),
            "n_items_human_verified": sum(1 for it in items if it.get("verified") is True),
            "papers": paper_info,
            "uploaded_as": [{"paper_id": p["paper_id"], "name": p["name"]}
                            for p in ingest["papers"]],
        },
        "config": {
            "k_values": list(cfg.k_values),
            "match_threshold": cfg.match_threshold,
            "judge_model": judge.model if judge else None,
        },
        "evidence_check": evidence,
        "summary": aggregate(records, cfg.k_values),
        "integrity": integrity,
        "results": records,
    }


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text or "dataset").strip("-")[:60] or "dataset"
