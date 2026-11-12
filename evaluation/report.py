"""Write evaluation results to JSON, CSV and a Markdown report."""

from __future__ import annotations

import csv
import json
import os
from typing import Optional

CSV_FIELDS_BASE = [
    "id", "question_type", "pipeline_status", "routes", "question",
    "n_relevant_sources", "n_retrieved_chunks", "n_context_chunks", "first_relevant_rank",
]
CSV_FIELDS_TAIL = [
    "context_recall", "faithfulness_llm_judge", "n_claims", "n_unsupported_claims",
    "n_contradicted_claims", "answer_relevance_llm_judge", "answer_relevance_rubric",
    "lexical_support", "unsupported_numbers", "key_fact_coverage", "embedding_relevance",
    "latency_retrieval_s", "latency_generation_s", "latency_other_s", "latency_total_s",
    "generation_model", "attempts", "errors", "answer", "expected_answer",
]


def _csv_value(v):
    if isinstance(v, (list, tuple)):
        return " | ".join(str(x) for x in v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else v


def write_csv(result: dict, path: str) -> None:
    ks = result["config"]["k_values"]
    fields = CSV_FIELDS_BASE + [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks] + CSV_FIELDS_TAIL
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in result["results"]:
            w.writerow({k: _csv_value(r.get(k)) for k in fields})


def write_json(result: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)


def _fmt(v: Optional[float], pct: bool = False, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.{digits}f}"


def _stat_row(label: str, s: dict, pct: bool = False, unit: str = "") -> str:
    if not s or s.get("n", 0) == 0:
        return f"| {label} | n/a | n/a | n/a | 0 |"
    f = (lambda v: _fmt(v, pct)) if pct else (lambda v: f"{v:.3f}{unit}")
    return f"| {label} | {f(s['mean'])} | {f(s['median'])} | {f(s['p95'])} | {s['n']} |"


def render_markdown(result: dict) -> str:
    s, ds, cfg, run = result["summary"], result["dataset"], result["config"], result["run"]
    ks = cfg["k_values"]
    lines = []
    a = lines.append
    a("# RAG evaluation report")
    a("")
    a(f"- **Run:** {run['started_utc']} → {run['finished_utc']} (UTC)")
    a(f"- **Code:** git `{run.get('git_commit', 'unknown')}`"
      + (" (uncommitted changes present)" if run.get("git_dirty") else ""))
    a(f"- **Dataset:** `{ds['name']}` — {ds['path']}")
    a(f"  - sha256 `{(ds.get('sha256') or 'n/a')[:16]}…`, status: **{ds.get('status') or 'unspecified'}**, "
      f"{ds['n_items_evaluated']} of {ds['n_items_total']} items evaluated, "
      f"{ds['n_items_human_verified']} marked human-verified")
    for p in ds["papers"]:
        flag = "" if p["sha256_matches"] else " ⚠️ sha256 differs from dataset"
        a(f"  - paper `{p['file']}` sha256 `{p['sha256'][:16]}…`{flag}")
    if run.get("generation_model_overridden"):
        a(f"- **Generation model: `{run['generation_model']}` — OVERRIDDEN for this evaluation run only.** "
          f"Production uses `{run.get('production_generation_model')}`; these answer-quality "
          "numbers do not describe the production configuration.")
    else:
        a(f"- **Generation model (production, unchanged):** `{run.get('generation_model', 'unknown')}`")
    a(f"- **Embedding model (production, unchanged):** `{run.get('embedding_model', 'unknown')}`")
    a(f"- **Judge model (LLM-as-a-judge):** `{cfg['judge_model'] or 'disabled'}`")
    a(f"- **Index:** {run['n_index_chunks']} chunks built in {run['index_build_seconds']:.1f}s "
      f"in an isolated workspace (session `{run['session_id']}`)")
    a("")
    if ds.get("status") != "verified" or ds["n_items_human_verified"] < ds["n_items_evaluated"]:
        a("> ⚠️ Some dataset items are not marked as human-verified. Expected answers and evidence "
          "quotes should be checked by someone who knows the papers before these numbers are cited.")
        a("")
    if result["summary"]["n_errors"]:
        a(f"> ⚠️ {result['summary']['n_errors']} question(s) failed inside the pipeline "
          "(see per-question table). They are excluded from quality and latency averages.")
        a("")

    a("## Summary")
    a("")
    a(f"- Questions evaluated: **{s['n_questions']}** "
      f"(answered by the LLM: {s['n_answered']}, no LLM call / refused: {s['n_no_generation']}, "
      f"pipeline errors: {s['n_errors']})")
    a("- Routes taken: " + ", ".join(f"`{k}` × {v}" for k, v in s["routes"].items()))
    a("")
    a("| Metric | Mean | Median | p95 | n |")
    a("|---|---|---|---|---|")
    for k in ks:
        a(_stat_row(f"Retrieval Recall@{k}", s[f"recall@{k}"], pct=True))
    for k in ks:
        a(_stat_row(f"Hit@{k} (≥1 relevant chunk)", s[f"hit@{k}"], pct=True))
    a(_stat_row("Context recall (relevant evidence in LLM prompt)", s["context_recall"], pct=True))
    a(_stat_row("Faithfulness — LLM judge", s["faithfulness_llm_judge"], pct=True))
    a(_stat_row("Answer relevance — LLM judge (rubric 1–5 → 0–1)", s["answer_relevance_llm_judge"], pct=True))
    a(_stat_row("Lexical support (supplementary)", s["lexical_support"], pct=True))
    a(_stat_row("Key-fact coverage (supplementary)", s["key_fact_coverage"], pct=True))
    a(_stat_row("Question↔answer embedding cosine (supplementary)", s["embedding_relevance"]))
    a(_stat_row("Retrieval latency", s["latency_retrieval_s"], unit=" s"))
    a(_stat_row("LLM generation latency", s["latency_generation_s"], unit=" s"))
    a(_stat_row("Total response latency (answered questions)", s["latency_total_s"], unit=" s"))
    a("")
    a(f"Answers containing numbers not found in their context: "
      f"**{s['n_answers_with_unsupported_numbers']}** of {s['n_answered']}.")
    a("")

    bad = [e for e in result["evidence_check"] if not e["ok"]]
    a("## Dataset evidence check")
    a("")
    if bad:
        a(f"{len(bad)} evidence quote(s) were not found in any indexed chunk "
          f"(coverage < {cfg['match_threshold']}); their Recall@K is 0 regardless of retrieval:")
        for e in bad:
            a(f"- `{e['item_id']}` source {e['source_index']} ({e['paper']}, page {e['page']}): "
              f"best coverage {e['best_coverage']}")
    else:
        a(f"All {len(result['evidence_check'])} evidence quotes were found in the index.")
    a("")

    a("## Per-question results")
    a("")
    head = "| id | status | route | " + " | ".join(f"R@{k}" for k in ks) + \
           " | ctx recall | faithful | relevance | retr s | gen s | total s |"
    a(head)
    a("|" + "---|" * (head.count("|") - 1))
    for r in result["results"]:
        a(f"| {r['id']} | {r['pipeline_status']} | {'>'.join(r['routes']) or '-'} | "
          + " | ".join(_fmt(r.get(f'recall@{k}'), pct=True) for k in ks)
          + f" | {_fmt(r.get('context_recall'), pct=True)} | {_fmt(r.get('faithfulness_llm_judge'), pct=True)}"
          f" | {_fmt(r.get('answer_relevance_llm_judge'), pct=True)}"
          f" | {_fmt(r.get('latency_retrieval_s'), digits=2)} | {_fmt(r.get('latency_generation_s'), digits=2)}"
          f" | {_fmt(r.get('latency_total_s'), digits=2)} |")
    a("")
    errs = [(r["id"], e) for r in result["results"] for e in r.get("errors", [])]
    errs += [(r["id"], r["answer"][:200]) for r in result["results"] if r["pipeline_status"] == "error"]
    if errs:
        a("### Errors")
        a("")
        for i, e in errs:
            a(f"- `{i}`: {e}")
        a("")

    integ = result["integrity"]
    a("## Data integrity")
    a("")
    if integ["unchanged"]:
        a("No file under the protected paths was added, removed or modified during the run:")
    else:
        a("⚠️ Files under the protected paths changed during the run "
          "(this can also happen if the app was used at the same time):")
        for kind in ("added", "removed", "modified"):
            for p in integ[kind]:
                a(f"- {kind}: `{p}`")
    for p in integ["protected_paths"]:
        a(f"- `{p}`")
    a("")

    a("## Methodology (short)")
    a("")
    a("- Each question is sent through the unmodified `rag_pipeline.ask_question`. FAISS search "
      "calls and Groq calls are observed from outside the module to capture retrieved chunks, the "
      "final prompt and timings.")
    a("- **Recall@K**: share of the item's evidence quotes covered by the first K chunks FAISS "
      f"returned (a chunk covers a quote if it is from the same paper/page and contains ≥ "
      f"{cfg['match_threshold']:.0%} of the quote's word trigrams). Questions without evidence are excluded.")
    a("- **Context recall**: the same check over only the chunks that ended up in the LLM prompt.")
    a("- **Faithfulness (LLM judge)**: the judge splits the answer into atomic claims and labels "
      "each supported / unsupported / contradicted using only the prompt's context. "
      "Score = supported ÷ all claims; unsupported claims count as failures.")
    a("- **Answer relevance (LLM judge)**: 1–5 rubric on whether the answer addresses the question "
      "(not whether it is correct), normalised to 0–1.")
    a("- **Latency**: retrieval = time inside FAISS search calls (incl. query embedding); "
      "generation = time inside Groq calls; total = wall time of `ask_question` "
      "(also includes index loading and, for metadata questions, PDF parsing).")
    a("- LLM-judge scores are model estimates, not ground truth. See `evaluation/README.md` for limitations.")
    a("")
    return "\n".join(lines)


def write_all(result: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "json": os.path.join(out_dir, "results.json"),
        "csv": os.path.join(out_dir, "results.csv"),
        "report": os.path.join(out_dir, "report.md"),
    }
    write_json(result, paths["json"])
    write_csv(result, paths["csv"])
    with open(paths["report"], "w", encoding="utf-8") as f:
        f.write(render_markdown(result))
    return paths
